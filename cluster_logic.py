import numpy as np
from sklearn.cluster import DBSCAN

def perform_clustering(time_matrix_seconds):
    """
    使用 DBSCAN 演算法對地點進行分群。

    Args:
        time_matrix_seconds (list of lists): 步行時間矩陣 (單位：秒)。

    Returns:
        list of lists: 分群結果，例如 [[0, 3], [1], [2]]，
                       其中數字是地點在原始列表中的索引。
    """
    # --- DBSCAN 參數設定 ---
    # eps: 兩個樣本被視為在彼此鄰域內的最大距離。
    # 我們的「距離」就是步行時間，15 分鐘 = 900 秒。
    eps_seconds = 15 * 60

    # min_samples: 一個點被視為核心點所需的鄰域樣本數。
    # 根據我們的定義，兩個點重疊即可成群，所以設為 2。
    min_samples = 2

    # 將我們的時間矩陣轉換為 NumPy 陣列，這是 scikit-learn 需要的格式
    matrix = np.array(time_matrix_seconds)

    # --- 執行 DBSCAN ---
    # metric='precomputed' 是最關鍵的一步！
    # 這告訴 DBSCAN 不要自己算距離，而是直接使用我們提供的矩陣作為距離。
    db = DBSCAN(eps=eps_seconds, min_samples=min_samples, metric='precomputed').fit(matrix)

    # --- 處理分群結果 ---
    labels = db.labels_  # 獲取每個點的群集標籤，-1 代表雜訊 (outlier)

    # This correctly counts clusters by ignoring the "noise" label (-1)
    n_clusters = len(set(label for label in labels if label != -1))
    print(f"預計分群數量: {n_clusters}")

    clusters = {i: [] for i in range(n_clusters)}
    # 將被標記為 -1 的雜訊點也視為獨立的單點群集
    noise_points_as_clusters = []

    for i, label in enumerate(labels):
        if label == -1:
            noise_points_as_clusters.append([i]) # 每個雜訊點自成一群
        else:
            clusters[label].append(i)

    # 合併正常的群集和雜訊點群集
    final_clusters = list(clusters.values()) + noise_points_as_clusters

    print(f"分群結果 (地點索引): {final_clusters}")
    return final_clusters