document.addEventListener('DOMContentLoaded', () => {
    const locationInput = document.getElementById('location-input');
    const addButton = document.getElementById('add-button');
    const locationList = document.getElementById('location-list');
    const planButton = document.getElementById('plan-button');
    const loadingSpinner = document.getElementById('loading-spinner');

    const addLocation = () => {
        const locationText = locationInput.value.trim();
        if (locationText) {
            const li = document.createElement('li');
            li.textContent = locationText;
            
            const removeButton = document.createElement('button');
            removeButton.textContent = '移除';
            removeButton.className = 'remove-button';
            removeButton.onclick = () => {
                locationList.removeChild(li);
            };
            
            li.appendChild(removeButton);
            locationList.appendChild(li);
            locationInput.value = '';
        }
    };

    addButton.addEventListener('click', addLocation);
    locationInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            addLocation();
        }
    });

    planButton.addEventListener('click', async () => {
        const locations = Array.from(locationList.querySelectorAll('li')).map(li => li.textContent.replace('移除', '').trim());
        const travelMode = document.querySelector('input[name="travel_mode"]:checked').value;

        if (locations.length < 2) {
            alert('請至少加入兩個地點！');
            return;
        }

        const requestBody = {
            locations: locations,
            travel_mode: travelMode
        };

        // 顯示載入動畫，禁用按鈕
        planButton.disabled = true;
        loadingSpinner.style.display = 'block';

        try {
            const response = await fetch('/plan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `伺服器錯誤: ${response.status}`);
            }

            const data = await response.json();
            
            // 使用 sessionStorage 在頁面間傳遞資料
            sessionStorage.setItem('planResult', JSON.stringify(data));
            
            // 跳轉到結果頁面
            window.location.href = '/results';

        } catch (error) {
            alert(`規劃失敗：${error.message}`);
        } finally {
            // 無論成功或失敗，都恢復按鈕狀態
            planButton.disabled = false;
            loadingSpinner.style.display = 'none';
        }
    });
});