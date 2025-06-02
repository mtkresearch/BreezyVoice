document.addEventListener('DOMContentLoaded', function() {
    const sftForm = document.getElementById('sftForm');
    const addSpeakerForm = document.getElementById('addSpeakerForm');
    const audioPlayer = document.getElementById('audioPlayer');
    const speakerSearch = document.getElementById('speakerSearch');
    let allSpeakers = []; // 存儲所有音色

    // 將PCM數據轉換為WAV格式
    function pcmToWav(pcmData) {
        const wavHeader = new ArrayBuffer(44);
        const view = new DataView(wavHeader);
        
        // RIFF identifier
        writeString(view, 0, 'RIFF');
        // RIFF chunk length
        view.setUint32(4, 36 + pcmData.byteLength, true);
        // RIFF type
        writeString(view, 8, 'WAVE');
        // format chunk identifier
        writeString(view, 12, 'fmt ');
        // format chunk length
        view.setUint32(16, 16, true);
        // sample format (raw)
        view.setUint16(20, 1, true);
        // channel count
        view.setUint16(22, 1, true);
        // sample rate
        view.setUint32(24, 22050, true);
        // byte rate (sample rate * block align)
        view.setUint32(28, 22050 * 2, true);
        // block align (channel count * bytes per sample)
        view.setUint16(32, 2, true);
        // bits per sample
        view.setUint16(34, 16, true);
        // data chunk identifier
        writeString(view, 36, 'data');
        // data chunk length
        view.setUint32(40, pcmData.byteLength, true);

        // 合併WAV頭部和PCM數據
        const wavData = new Blob([wavHeader, pcmData], { type: 'audio/wav' });
        return wavData;
    }

    // 輔助函數：寫入字符串到DataView
    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    // SFT 模式表單提交
    sftForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const form = e.target;
        form.classList.add('loading');

        try {
            const formData = new FormData();
            formData.append('tts_text', document.getElementById('sftText').value);
            formData.append('spk_id', document.getElementById('sftSpeaker').value);

            const response = await fetch('/inference_sft', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('生成失敗');

            const pcmData = await response.arrayBuffer();
            const wavData = pcmToWav(pcmData);
            const audioUrl = URL.createObjectURL(wavData);
            audioPlayer.src = audioUrl;
            audioPlayer.play();
        } catch (error) {
            alert('生成語音時發生錯誤：' + error.message);
        } finally {
            form.classList.remove('loading');
        }
    });

    // 新增音色表單提交
    addSpeakerForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const form = e.target;
        form.classList.add('loading');

        try {
            const formData = new FormData();
            formData.append('spk_id', document.getElementById('speakerId').value);
            formData.append('prompt_wav', document.getElementById('speakerAudio').files[0]);
            
            const speakerText = document.getElementById('speakerText').value;
            if (speakerText) {
                formData.append('prompt_text', speakerText);
            }

            const response = await fetch('/add_speaker', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                alert('成功新增音色！');
                // 重新載入音色列表
                loadSpeakers();
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            alert('新增音色時發生錯誤：' + error.message);
        } finally {
            form.classList.remove('loading');
        }
    });

    // 刪除音色
    async function removeSpeaker(spkId) {
        if (!confirm(`確定要刪除音色 ${spkId} 嗎？`)) {
            return;
        }

        try {
            const formData = new FormData();
            formData.append('spk_id', spkId);

            const response = await fetch('/remove_speaker', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                alert('成功刪除音色！');
                // 重新載入音色列表
                loadSpeakers();
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            alert('刪除音色時發生錯誤：' + error.message);
        }
    }

    // 更新音色列表UI
    function updateSpeakerList(speakers) {
        const speakerList = document.getElementById('speakerList');
        const speakerSelect = document.getElementById('sftSpeaker');
        
        // 清空現有列表
        speakerList.innerHTML = '';
        speakerSelect.innerHTML = '<option value="">請選擇音色...</option>';
        
        // 添加新的選項
        speakers.forEach(speaker => {
            // 添加到下拉選單
            const option = document.createElement('option');
            option.value = speaker;
            option.textContent = speaker;
            speakerSelect.appendChild(option);

            // 添加到表格
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${speaker}</td>
                <td>
                    <button class="btn btn-danger btn-sm" onclick="removeSpeaker('${speaker}')">刪除</button>
                </td>
            `;
            speakerList.appendChild(row);
        });
    }

    // 搜尋音色
    function searchSpeakers(query) {
        const filteredSpeakers = allSpeakers.filter(speaker => 
            speaker.toLowerCase().includes(query.toLowerCase())
        );
        updateSpeakerList(filteredSpeakers);
    }

    // 設置搜尋事件監聽器
    speakerSearch.addEventListener('input', function(e) {
        searchSpeakers(e.target.value);
    });

    // 載入音色列表
    async function loadSpeakers() {
        try {
            const response = await fetch('/get_speakers');
            if (!response.ok) throw new Error('獲取音色列表失敗');
            
            allSpeakers = await response.json();
            const searchQuery = speakerSearch.value;
            searchSpeakers(searchQuery);
        } catch (error) {
            console.error('載入音色列表失敗：', error);
        }
    }

    // 將removeSpeaker函數添加到全局作用域
    window.removeSpeaker = removeSpeaker;

    // 初始載入音色列表
    loadSpeakers();
}); 