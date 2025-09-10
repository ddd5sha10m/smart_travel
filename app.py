import os
import googlemaps
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from cluster_logic import perform_clustering
# 匯入我們新建立的路徑規劃函式
from routing_logic import calculate_cluster_matrix, solve_cluster_tsp_bruteforce, create_final_itinerary, solve_points_tsp_with_ga
# --- 1. 初始化設定 (與之前相同) ---
app = Flask(__name__)
load_dotenv()
api_key = os.getenv("GOOGLE_MAPS_API_KEY")
if not api_key:
    raise ValueError("未找到 Google Maps API 金鑰，請檢查您的 .env 檔案")
gmaps = googlemaps.Client(key=api_key)

# --- 2. 核心功能函式 (與之前相同) ---
def get_walking_time_matrix(locations):
    """
    計算給定地點列表之間的步行時間矩陣。
    (V3 - 修正：用極大值代替 -1 來處理無法到達的路徑)
    """
    num_locations = len(locations)
    # 定義一個極大值來表示「無限遠」或「不可達」
    # 使用整數比 float('inf') 對某些函式庫更安全
    UNREACHABLE = 9999999 

    # 先初始化一個 N x N 的矩陣，填滿我們的「不可達」值
    time_matrix = [[UNREACHABLE] * num_locations for _ in range(num_locations)]

    for i in range(num_locations):
        origin = [locations[i]]
        destinations = locations

        try:
            matrix_result = gmaps.distance_matrix(origins=origin,
                                                  destinations=destinations,
                                                  mode="walking")
            
            row_elements = matrix_result['rows'][0]['elements']
            
            for j in range(len(row_elements)):
                if row_elements[j]['status'] == 'OK':
                    duration_seconds = row_elements[j]['duration']['value']
                    time_matrix[i][j] = duration_seconds
                else:
                    # 如果路徑不存在，矩陣中會保留 UNREACHABLE 值
                    # 我們依然可以印出警告，但程式不會再用 -1
                    print(f"警告: 無法計算從 '{locations[i]}' 到 '{locations[j]}' 的路徑。")
            
            # 一個地點到自己的距離應為 0
            time_matrix[i][i] = 0

        except Exception as e:
            print(f"呼叫 Google API 時發生錯誤 (處理起點 '{locations[i]}'): {e}")
            continue
            
    return time_matrix


@app.route('/')
def index():
    """提供主頁面 (輸入頁)"""
    return render_template('index.html')

@app.route('/results')
def results():
    """提供結果頁面"""
    # 我們需要將 API 金鑰傳遞給前端，以便 Google Maps JavaScript API 使用
    google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    return render_template('results.html', api_key=google_maps_api_key)

# --- 3. 更新後的 API 路由 ---
@app.route('/plan', methods=['POST'])
def plan_route_api():
    data = request.get_json()
    if not data or 'locations' not in data:
        return jsonify({"error": "請求中缺少 'locations' 列表"}), 400
    locations = data['locations']
    
    allowed_modes = ['driving', 'transit', 'bicycling']
    travel_mode = data.get('travel_mode', 'driving')
    
    if travel_mode not in allowed_modes:
        return jsonify({"error": f"不支援的 travel_mode: '{travel_mode}'。請使用 {allowed_modes} 中的一個。"}), 400
    
    # 步驟 1: 計算步行矩陣
    walking_matrix = get_walking_time_matrix(locations)
    if not walking_matrix:
        return jsonify({"error": "無法計算步行時間矩陣"}), 500

    # 步驟 2: 分群
    clusters = perform_clustering(walking_matrix)
    
    # 處理單一群集的情況
    if not clusters:
        return jsonify({"error": "無法形成任何群集"}), 500
    if len(clusters) == 1:
        single_cluster_indices = clusters[0]
        final_route_indices = solve_points_tsp_with_ga(single_cluster_indices, walking_matrix)
        final_route_locations = [locations[i] for i in final_route_indices]
        response = {
            "message": "所有地點都在同一個步行區域內，已優化內部步行路線。",
            "travel_mode_used": "walking",
            "locations": locations,
            "clusters": clusters,
            "final_itinerary_indices": final_route_indices,
            "final_itinerary_locations": final_route_locations
        }
        return jsonify(response)

    # 步驟 3: 完整多群集規劃
    # 3a: 計算群集矩陣
    cluster_travel_matrix = calculate_cluster_matrix(gmaps, locations, clusters, mode=travel_mode)
    if cluster_travel_matrix is None:
        return jsonify({"error": "無法計算群集交通矩陣"}), 500

    # 3b: 基因演算法找出最佳群集順序
    optimal_cluster_order = solve_cluster_tsp_bruteforce(cluster_travel_matrix)
    
    # 3c: 使用我們重構後的函式，產生最終的精確行程
    final_route_indices = create_final_itinerary(locations, clusters, optimal_cluster_order, walking_matrix, gmaps, mode=travel_mode)
    final_route_locations = [locations[i] for i in final_route_indices]
    
    # 組合最終的回應
    response = {
        "travel_mode_used": travel_mode,
        "locations": locations,
        "clusters": clusters,
        "optimal_cluster_order": optimal_cluster_order,
        "final_itinerary_indices": final_route_indices,
        "final_itinerary_locations": final_route_locations,
    }
    return jsonify(response)

# --- 4. 啟動伺服器 (與之前相同) ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)