import open3d as o3d
import numpy as np


type = 1


if type==1:
    # PCD 파일 로드
    static_pcd = o3d.io.read_point_cloud("data/parkinglot/01/outputs2/static_points.pcd")
    dynamic_pcd = o3d.io.read_point_cloud("data/parkinglot/01/outputs2/dynamic_points.pcd")
    total_point = static_pcd + dynamic_pcd

    # 시각화
    o3d.visualization.draw_geometries([total_point])

if type==2:

    pcd = o3d.io.read_point_cloud("data/parkinglot/02/outputs/lifelong_map.pcd")
    o3d.visualization.draw_geometries([pcd])
    
if type==3:
    
    data = np.load("data/parkinglot/01/outputs1/noneffect_points.npy")
    print(f"noneffect points : {data.shape}")
    
    data = np.load("data/parkinglot/01/outputs1/total_points.npy")
    print(f"total points : {data.shape}")