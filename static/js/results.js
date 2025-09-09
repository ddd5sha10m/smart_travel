function initMap() {
    // 從 sessionStorage 獲取結果
    const planResultString = sessionStorage.getItem('planResult');
    if (!planResultString) {
        alert('找不到規劃結果，請返回主頁重新規劃。');
        window.location.href = '/';
        return;
    }
    const planResult = JSON.parse(planResultString);
    const itineraryList = document.getElementById('itinerary-list');
    const finalRoute = planResult.final_itinerary_locations;

    if (!finalRoute || finalRoute.length === 0) {
        itineraryList.innerHTML = '<li>無法產生路線。</li>';
        return;
    }
    
    // 填充路線列表
    finalRoute.forEach(location => {
        const li = document.createElement('li');
        li.textContent = location;
        itineraryList.appendChild(li);
    });

    // 初始化地圖
    const map = new google.maps.Map(document.getElementById('map'), {
        zoom: 14,
        center: { lat: 25.033, lng: 121.565 }, // 預設中心點 (台北)
        mapTypeControl: false,
    });
    
    const directionsService = new google.maps.DirectionsService();
    const directionsRenderer = new google.maps.DirectionsRenderer({
        map: map,
        suppressMarkers: true, // 我們要自訂標記
    });
    
    // 建立路線請求
    const origin = finalRoute[0];
    const destination = finalRoute[finalRoute.length - 1];
    const waypoints = finalRoute.slice(1, -1).map(location => ({
        location: location,
        stopover: true,
    }));
    
    // 將 travel_mode 轉換為 Google Maps API 的 TravelMode
    let travelMode = planResult.travel_mode_used.toUpperCase();
    if(travelMode === 'BICYCLING') travelMode = 'BICYCLING'; // API ENUM is BICYCLING
    else if(travelMode === 'TRANSIT') travelMode = 'TRANSIT';
    else travelMode = 'DRIVING';


    directionsService.route(
        {
            origin: origin,
            destination: destination,
            waypoints: waypoints,
            travelMode: google.maps.TravelMode[travelMode],
            // 為了畫出步行路線，我們這裡統一用 DRIVING/TRANSIT/BICYCLING 呈現
            // 更複雜的應用可以分段請求
        },
        (response, status) => {
            if (status === 'OK') {
                directionsRenderer.setDirections(response);
                
                // 創建自訂的數字標記
                const route = response.routes[0];
                let legStartLocation;
                // Origin marker
                addMarker(route.legs[0].start_location, '1', '起點');
                legStartLocation = route.legs[0].start_location;

                // Waypoint and Destination markers
                for (let i = 0; i < route.legs.length; i++) {
                    const leg = route.legs[i];
                    const labelIndex = (i + 2).toString();
                    const title = finalRoute[i+1];
                    addMarker(leg.end_location, labelIndex, title);
                }
                
            } else {
                window.alert('繪製路線失敗: ' + status);
                // 即使路線繪製失敗，也要放上基本標記
                addBasicMarkers();
            }
        }
    );
    
    function addMarker(position, label, title) {
        new google.maps.Marker({
            position,
            map,
            label: {
                text: label,
                color: "white"
            },
            title,
        });
    }

    function addBasicMarkers() {
        // ... (備用方案，這裡省略) ...
    }
}