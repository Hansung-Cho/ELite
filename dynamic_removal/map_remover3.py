import os
import yaml
import numpy as np
import open3d as o3d
from tqdm import trange
from scipy.spatial import KDTree
import psutil, os
import time

from utils.session import Session
from utils.session_map import SessionMap
from utils.logger import logger
import heapq


class MapRemover:
    def __init__(
        self, 
        config_path: str
    ):
        # Load parameters
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)

        p_settings = self.params["settings"]        
        os.makedirs(p_settings["output_dir"], exist_ok=True)
        
        self.std_dev_o = 0.025
        self.std_dev_f = 0.025
        self.alpha = 0.5
        self.beta = 0.1

        self.session_loader : Session = None
        self.session_map : SessionMap = None


    def load(self, new_session : Session = None):
        p_settings = self.params["settings"]

        if new_session is None:
            self.session_loader = Session(p_settings["scans_dir"], p_settings["poses_file"])
        else:
            self.session_loader = new_session
        
        logger.info(f"Loaded new session")


    def run(self):
        p_settings = self.params["settings"]
        p_dor = self.params["dynamic_object_removal"]

        # 1) Aggregate scans to create session map
        session_map = self.session_loader[0:len(self.session_loader)].downsample(0.01).get()
        eph_l = np.zeros(len(session_map.points))
        logger.info(f"Initialized session map")

        # 2) Select anchor points for local ephemerality update
        anchor_points = session_map.voxel_down_sample(p_dor["anchor_voxel_size"])
        anchor_eph_l = np.ones(len(anchor_points.points)) * 0.5 # initial value
        anchor_kdtree = KDTree(np.asarray(anchor_points.points))

        threshold = np.sqrt(self.std_dev_f * np.log(self.alpha/self.beta))

        noneffect_cache = {} 

        def clean_cache_by_pose_distance(noneffect_cache, current_pose, max_cache_size=10_000_000, del_perc = 0.1):
            if len(noneffect_cache) > max_cache_size:
                num_to_remove = int(len(noneffect_cache) * del_perc) 

                farthest_keys = heapq.nlargest(
                    num_to_remove,
                    noneffect_cache.items(),
                    key=lambda item: np.linalg.norm(item[1]["position"] - current_pose)
                )

                for key, _ in farthest_keys:
                    del noneffect_cache[key]        

        logger.info(f"Updating anchor local ephemerality")
        for i in trange(0, len(self.session_loader), p_dor["stride"], desc="Updating \u03B5_l", ncols=100):

            logger.debug(f"Processing scan {i + 1}/{len(self.session_loader)}")
            scan = np.asarray(self.session_loader[i].get().points)
            pose = self.session_loader.get_pose(i)[:3, 3]
            
            # occupied space update
            dists, inds = anchor_kdtree.query(scan, k=p_dor["num_k"])
            for j in range(len(dists)):
                dist = dists[j]
                eph_l_prev = anchor_eph_l[inds[j]]
                update_rate = np.minimum(self.alpha * (1 - np.exp(-1 * dist**2 / self.std_dev_o)) + self.beta, self.alpha) # Eq. 5 
                eph_l_new = eph_l_prev * update_rate / (
                    eph_l_prev * update_rate + (1 - eph_l_prev) * (1 - update_rate)
                )
                anchor_eph_l[inds[j]] = eph_l_new

            # free space update
            shifted_scan = scan - pose # local coordinates
            sample_ratios = np.linspace(p_dor["min_ratio"], p_dor["max_ratio"], p_dor["num_samples"])
            free_space_samples = pose + shifted_scan[:, np.newaxis, :] * sample_ratios.T[np.newaxis, :, np.newaxis]
            free_space_samples = free_space_samples.reshape(-1, 3)
            free_space_samples_o3d = o3d.geometry.PointCloud()
            free_space_samples_o3d.points = o3d.utility.Vector3dVector(free_space_samples)
            free_space_samples_o3d = free_space_samples_o3d.voxel_down_sample(voxel_size=p_dor["free_voxel_size"])
            free_space_samples = np.asarray(free_space_samples_o3d.points)
            
            # 파라미터
            batch_size = 20000  # 1~5만 사이로 조정 가능

            cache_miss = 0
            cache_noneffect_hit = 0

            # 1. 캐시 확인: hit이면 바로 업데이트, miss이면 모아서 batch query
            pts_to_query = []
            keys_to_query = []

            def to_voxel_key(point, voxel_size):
                return tuple(np.floor(point / voxel_size).astype(int))    

            for pt in free_space_samples:
                key = to_voxel_key(pt, p_dor["free_voxel_size"])
                if key in noneffect_cache:
                    cache_noneffect_hit += 1
                    inds = noneffect_cache[key]["inds"]
                    for ind in inds:
                        eph_l_prev = anchor_eph_l[ind]
                        update_rate = self.alpha
                        eph_l_new = eph_l_prev * update_rate / (
                            eph_l_prev * update_rate + (1 - eph_l_prev) * (1 - update_rate)
                        )
                        anchor_eph_l[ind] = eph_l_new
                else:
                    cache_miss += 1
                    pts_to_query.append(pt)
                    keys_to_query.append(key)

            # 2. Batch query for nonexisting point in dictionary
            free_start_time = time.perf_counter()

            if len(pts_to_query) > 0:
                pts_to_query = np.array(pts_to_query)
                num_pts = len(pts_to_query)

                for start in range(0, num_pts, batch_size):
                    end = min(start + batch_size, num_pts)
                    pts_chunk = pts_to_query[start:end]
                    keys_chunk = keys_to_query[start:end]

                    dists_chunk, inds_chunk = anchor_kdtree.query(
                        pts_chunk, k=p_dor["num_k"]
                    )

                    for key, pt, dists, inds in zip(keys_chunk, pts_chunk, dists_chunk, inds_chunk):
                        dmin = np.min(dists)

                        if dmin > threshold:
                            noneffect_cache[key] = {
                                "inds": inds,
                                "position": pt
                            }
                            # Deleting farthest point based on current location
                            clean_cache_by_pose_distance(noneffect_cache, pose)

                            inds = np.atleast_1d(inds)
                            for i in inds:
                                eph_l_prev = anchor_eph_l[i]
                                update_rate = self.alpha
                                eph_l_new = eph_l_prev * update_rate / (
                                    eph_l_prev * update_rate + (1 - eph_l_prev) * (1 - update_rate)
                                )
                                anchor_eph_l[i] = eph_l_new
                        else:
                            for dist, ind in zip(dists, inds):
                                for i in np.atleast_1d(ind):
                                    eph_l_prev = anchor_eph_l[i]
                                    update_rate = np.maximum(
                                        self.alpha * (1 + np.exp(-1 * dist**2 / self.std_dev_f)) - self.beta,
                                        self.alpha,
                                    )
                                    eph_l_new = eph_l_prev * update_rate / (
                                        eph_l_prev * update_rate + (1 - eph_l_prev) * (1 - update_rate)
                                    )
                                    anchor_eph_l[i] = eph_l_new

            free_end_time = time.perf_counter()

            # 3. Logging
            total = cache_miss + cache_noneffect_hit
            if total > 0:
                logger.debug(
                    f"noneffect_hit={cache_noneffect_hit/total:.1%}, "
                    f"miss={cache_miss/total:.1%}"
                )
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / (1024**2)
                logger.info(
                    f"Cache size : noneffect={len(noneffect_cache)}  |  Memory usage: {mem_mb:.2f} MB"
                )
                logger.info(
                    f"Free space update time: {free_end_time - free_start_time:.2f} s"
                )
                logger.info("")
                
                
        noneffect_cache.clear()


        # 3) Propagate anchor local ephemerality to session map
        distances, indices = anchor_kdtree.query(np.asarray(session_map.points), k=p_dor["num_k"])
        distances = np.maximum(distances, 1e-6) # avoid division by zero
        weights = 1 / (distances**2)
        weights /= np.sum(weights, axis=1, keepdims=True)
        eph_l = np.sum(weights * anchor_eph_l[indices], axis=1)
        eph_l = np.clip(eph_l, 0, 1) # redundant, but for safety

        # 4) Remove dynamic objects to create cleaned session map
        static_points = session_map.select_by_index(np.where(eph_l <= p_dor["dynamic_threshold"])[0])
        static_eph_l = eph_l[eph_l <= p_dor["dynamic_threshold"]]
        static_points.paint_uniform_color([0.5, 0.5, 0.5])
        dynamic_points = session_map.select_by_index(np.where(eph_l > p_dor["dynamic_threshold"])[0])
        dynamic_points.paint_uniform_color([1, 0, 0])
                  
        if p_dor["save_static_dynamic_map"]:
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "static_points.pcd"), static_points)  
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "dynamic_points.pcd"), dynamic_points)
        if p_dor["viz_static_dynamic_map"]:
            total_points = static_points + dynamic_points
            o3d.visualization.draw_geometries([total_points])

        cleaned_session_map = SessionMap(
            np.asarray(static_points.points), static_eph_l
        )
        self.session_map = cleaned_session_map

        if p_dor["save_cleaned_session_map"]:
            cleaned_session_map.save(p_settings["output_dir"], is_global=False) 
        if p_dor["viz_cleaned_session_map"]:
            cleaned_session_map.visualize()

        return cleaned_session_map


    def get(self):
        return self.session_map
        

# Example usage
if __name__ == "__main__":
    config = "../config/sample.yaml"
    remover = MapRemover(config)
    # Load session using the config file or from an alingment module
    remover.load()
    # Run the dynamic object removal
    remover.run()
    # Get the cleaned session map
    cleaned_session_map = remover.get()