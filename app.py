import os
import googlemaps
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from cluster_logic import perform_clustering
# 匯入我們新建立的路徑規劃函式
from routing_logic import calculate_cluster_matrix, solve_cluster_tsp_with_ga, create_final_itinerary
# --- 1. 初始化設定 (與之前相同) ---
app = Flask(__name__)
load_dotenv()
api_key = os.getenv("GOOGLE_MAPS_API_KEY")
if not api_key:
    raise ValueError("未找到 Google Maps API 金鑰，請檢查您的 .env 檔案")
gmaps = googlemaps.Client(key=api_key)

# --- 2. 核心功能函式 (與之前相同) ---
def get_walking_time_matrix(locations):
    # ... (程式碼與之前完全相同，此處省略) ...
    try:
        matrix = gmaps.distance_matrix(origins=locations,
                                       destinations=locations,
                                       mode="walking",
                                       language="zh-TW")
    except Exception as e:
        print(f"呼叫 Google API 時發生錯誤: {e}")
        return None
    num_locations = len(locations)
    time_matrix = [[-1] * num_locations for _ in range(num_locations)]
    for i in range(num_locations):
        for j in range(num_locations):
            if matrix['rows'][i]['elements'][j]['status'] == 'OK':
                duration_seconds = matrix['rows'][i]['elements'][j]['duration']['value']
                time_matrix[i][j] = duration_seconds
            else:
                print(f"警告: 無法計算從 '{locations[i]}' 到 '{locations[j]}' 的路徑。")
    return time_matrix

# --- 3. 更新後的 API 路由 ---
@app.route('/plan', methods=['POST'])
def plan_route_api():
    data = request.get_json()
    if not data or 'locations' not in data:
        return jsonify({"error": "請求中缺少 'locations' 列表"}), 400
    locations = data['locations']
    
    # 步驟 1: 計算步行時間矩陣
    walking_matrix = get_walking_time_matrix(locations)
    if not walking_matrix:
        return jsonify({"error": "無法計算步行時間矩陣"}), 500

    # 步驟 2: 執行地點分群
    clusters = perform_clustering(walking_matrix)
    if not clusters or len(clusters) == 1:
        # 如果只有一個群集，就不需要進行群集間的規劃
        return jsonify({
            "locations": locations,
            "walking_time_matrix_seconds": walking_matrix,
            "clusters": clusters
        })

    # 步驟 3a (新): 執行宏觀路徑規劃
    # 3a.1: 計算群集之間的交通時間矩陣 (這裡以開車為例)
    cluster_driving_matrix = calculate_cluster_matrix(gmaps, locations, clusters, mode='driving')
    if cluster_driving_matrix is None:
        return jsonify({"error": "無法計算群集交通矩陣"}), 500

    # 3a.2: 使用基因演算法找出最佳群集順序
    optimal_cluster_order = solve_cluster_tsp_with_ga(cluster_driving_matrix)
    final_route_indices = create_final_itinerary(locations, clusters, optimal_cluster_order, walking_matrix, gmaps)

    # 將索引轉換回地點名稱，方便閱讀
    final_route_locations = [locations[i] for i in final_route_indices]
    
    # 組合最終的回應
    response = {
        "locations": locations,
        "clusters": clusters,
        "optimal_cluster_order": optimal_cluster_order,
        "final_itinerary_indices": final_route_indices,
        "final_itinerary_locations": final_route_locations,
        # (也可以保留矩陣資訊，方便除錯)
        # "walking_time_matrix_seconds": walking_matrix,
        # "cluster_driving_matrix_seconds": cluster_driving_matrix.tolist(),
    }
    return jsonify(response)

# --- 4. 啟動伺服器 (與之前相同) ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)