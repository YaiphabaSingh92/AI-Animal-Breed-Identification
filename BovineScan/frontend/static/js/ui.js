document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const previewImg = document.getElementById("preview-img");
    const clearImgBtn = document.getElementById("clear-img");
    const analyzeBtn = document.getElementById("analyze-btn");
    const loader = document.getElementById("loader");
    const resultBox = document.getElementById("result-box");
    const resultBreed = document.getElementById("result-breed");
    const resultConf = document.getElementById("result-conf");
    
    let currentFile = null;

    // --- CUSTOM ALERT LOGIC ---
    const customAlertOverlay = document.getElementById("custom-alert");
    const customAlertText = document.getElementById("custom-alert-text");
    const customAlertBtn = document.getElementById("custom-alert-btn");

    function showCustomAlert(message) {
        customAlertText.textContent = message;
        customAlertOverlay.classList.add("active");
    }

    if (customAlertBtn) {
        customAlertBtn.addEventListener("click", () => {
            customAlertOverlay.classList.remove("active");
        });
    }

    // --- RECENT SCANS LOGIC (PERSISTENT) ---
    const RECENT_SCANS_KEY = "bovinescan_persistent_scans";
    let recentScans = [];
    
    try {
        recentScans = JSON.parse(localStorage.getItem(RECENT_SCANS_KEY) || "[]");
    } catch(e) {
        recentScans = [];
    }

    function renderAllRecent() {
        const recentSection = document.getElementById("recent-section");
        const recentGrid = document.getElementById("recent-grid");
        
        recentGrid.innerHTML = "";
        
        if (recentScans.length > 0) {
            recentSection.style.display = "block";
        } else {
            recentSection.style.display = "none";
        }
        
        recentScans.forEach(scan => {
            const card = document.createElement("div");
            card.className = "recent-card";
            card.innerHTML = `
                <img class="recent-img" src="${scan.imgSrc}" alt="${scan.breed}">
                <div class="recent-info">
                    <div class="recent-info-breed">${scan.breed}</div>
                    <div class="recent-info-conf">${scan.conf}%</div>
                </div>
            `;
            recentGrid.appendChild(card);
        });
    }

    // Render initially from LocalStorage
    renderAllRecent();

    function createThumbnail(imageSrc, callback) {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement("canvas");
            const MAX_SIZE = 250; 
            let width = img.width;
            let height = img.height;
            
            if (width > height) {
                if (width > MAX_SIZE) {
                    height *= MAX_SIZE / width;
                    width = MAX_SIZE;
                }
            } else {
                if (height > MAX_SIZE) {
                    width *= MAX_SIZE / height;
                    height = MAX_SIZE;
                }
            }
            
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0, width, height);
            
            // Highly compressed base64 JPEG to bypass localStorage quotas
            callback(canvas.toDataURL("image/jpeg", 0.6));
        };
        img.src = imageSrc;
    }

    function addRecentInference(imgSrc, breed, conf) {
        createThumbnail(imgSrc, (thumbDataUrl) => {
            const scan = { imgSrc: thumbDataUrl, breed, conf };
            recentScans.unshift(scan);
            if (recentScans.length > 8) {
                recentScans.pop();
            }
            try {
                localStorage.setItem(RECENT_SCANS_KEY, JSON.stringify(recentScans));
            } catch (e) {
                console.warn("Storage cap hit. Truncating to save space.");
                recentScans = [scan];
                localStorage.setItem(RECENT_SCANS_KEY, JSON.stringify(recentScans));
            }
            renderAllRecent();
        });
    }

    // --- UPLOAD INTERFACE LOGIC ---

    dropZone.addEventListener("click", (e) => {
        if (e.target !== clearImgBtn && !clearImgBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    clearImgBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        clearFile();
    });

    function handleFile(file) {
        if (!file.type.startsWith("image/")) {
            showCustomAlert("Please upload an image file.");
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            dropZone.classList.add("has-image");
        };
        reader.readAsDataURL(file);
        
        resultBox.classList.remove("active");
    }

    function clearFile() {
        currentFile = null;
        fileInput.value = "";
        previewImg.src = "";
        dropZone.classList.remove("has-image");
        resultBox.classList.remove("active");
    }

    // --- INFERENCE API REQUEST ---

    analyzeBtn.addEventListener("click", async () => {
        if (!currentFile) {
            showCustomAlert("Please upload an image first.");
            return;
        }

        analyzeBtn.style.display = "none";
        loader.style.display = "block";
        resultBox.classList.remove("active");

        const formData = new FormData();
        formData.append("file", currentFile);

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (data.success && data.data) {
                let breedString = data.data.breed.replace(/_/g, " ");
                resultBreed.textContent = breedString;
                resultConf.textContent = `Confidence: ${data.data.confidence}%`;
                
                // Build the chart
                const chartContainer = document.getElementById("result-chart");
                chartContainer.innerHTML = "<div class='chart-title'>Top 5 Predictions</div>";
                
                data.data.top_5.forEach(item => {
                    const row = document.createElement("div");
                    row.className = "chart-row";
                    
                    const label = document.createElement("span");
                    label.className = "chart-label";
                    label.textContent = item.breed;
                    
                    const barBg = document.createElement("div");
                    barBg.className = "chart-bar-bg";
                    
                    const barFill = document.createElement("div");
                    barFill.className = "chart-bar-fill";
                    barFill.style.width = Math.max(item.confidence, 1) + "%";
                    
                    const val = document.createElement("span");
                    val.className = "chart-val";
                    val.textContent = item.confidence + "%";
                    
                    barBg.appendChild(barFill);
                    row.appendChild(label);
                    row.appendChild(barBg);
                    row.appendChild(val);
                    chartContainer.appendChild(row);
                });
                
                resultBox.classList.add("active");
                
                // Save to LocalStorage properly
                addRecentInference(previewImg.src, breedString, data.data.confidence);
            } else {
                showCustomAlert("Error during classification: " + (data.error || "Unknown error"));
            }
        } catch (err) {
            console.error(err);
            showCustomAlert("Network error. Please try again.");
        } finally {
            analyzeBtn.style.display = "block";
            loader.style.display = "none";
        }
    });
});
