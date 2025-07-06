document.addEventListener('DOMContentLoaded', () => {
    // --- 要素の取得 ---
    const uploadButton = document.getElementById('upload-button');
    const fileInput = document.getElementById('file-input');
    const imageEditor = document.querySelector('.image-editor');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const brushSizeSlider = document.getElementById('brush-size');
    const eraseButton = document.getElementById('erase-button');
    const downloadButton = document.getElementById('download-button');
    const resultImageDiv = document.querySelector('.result-image');
    const processedImage = document.getElementById('processed-image');

    // --- 変数の初期化 ---
    let isDrawing = false;
    let originalImage = null; // 元の画像(Imageオブジェクト)を保持する変数
    let lastX = 0;
    let lastY = 0;

    // --- イベントリスナー ---

    // 「画像をアップロード」ボタン
    uploadButton.addEventListener('click', () => {
        fileInput.click();
    });

    // ファイル選択
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            const image = new Image();
            image.onload = () => {
                // 元の画像を保存
                originalImage = image;

                // エディターを表示し、Canvasをセットアップ
                imageEditor.style.display = 'flex';
                resultImageDiv.style.display = 'none'; // 前回の結果を隠す
                downloadButton.style.display = 'none'; // 前回のダウンロードボタンを隠す

                // Canvasのサイズを元の画像と同じにする
                canvas.width = image.width;
                canvas.height = image.height;
                console.log('Canvas size (original image dimensions):', canvas.width, canvas.height);
                
                // Canvasに画像を描画
                ctx.drawImage(image, 0, 0, image.width, image.height);

                // ブラシの初期設定
                ctx.lineWidth = brushSizeSlider.value;
                ctx.lineCap = 'round';
                ctx.strokeStyle = 'red';
            };
            image.src = event.target.result;
        };
        reader.readAsDataURL(file);
    });

    // ブラシサイズスライダー
    brushSizeSlider.addEventListener('input', (e) => {
        ctx.lineWidth = e.target.value;
    });

    // 「削除」ボタン
    eraseButton.addEventListener('click', async () => {
        if (!originalImage) {
            alert("画像がアップロードされていません。");
            return;
        }

        // 元の画像をBlobに変換
        const originalImageBlob = await (await fetch(originalImage.src)).blob();

        // Canvas(マスク)をBlobに変換
        canvas.toBlob(async (maskBlob) => {
            const formData = new FormData();
            formData.append('image', originalImageBlob, 'image.png');
            formData.append('mask', maskBlob, 'mask.png');

            // バックエンドに送信
            try {
                const response = await fetch('/erase', {
                    method: 'POST',
                    body: formData,
                });

                if (response.ok) {
                    const data = await response.json();
                    processedImage.src = data.image;
                    resultImageDiv.style.display = 'block';
                    downloadButton.style.display = 'inline-block';
                } else {
                    alert('画像の処理に失敗しました。');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('エラーが発生しました。');
            }
        }, 'image/png');
    });

    // 「ダウンロード」ボタン
    downloadButton.addEventListener('click', () => {
        const link = document.createElement('a');
        link.download = 'erased-image.png';
        link.href = processedImage.src;
        link.click();
    });

    // --- マウス描画処理 ---
    const startDrawing = (e) => {
        isDrawing = true;
        const pos = getMousePos(canvas, e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        lastX = pos.x;
        lastY = pos.y;
    };

    const draw = (e) => {
        if (!isDrawing) return;
        const pos = getMousePos(canvas, e);

        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);

        lastX = pos.x;
        lastY = pos.y;
    };

    const stopDrawing = () => {
        isDrawing = false;
        ctx.closePath();
    };

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);

    function getMousePos(canvas, evt) {
        const rect = canvas.getBoundingClientRect();
        // Canvasの表示サイズと実際のピクセル寸法の比率を考慮
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (evt.clientX - rect.left) * scaleX,
            y: (evt.clientY - rect.top) * scaleY
        };
    }
});