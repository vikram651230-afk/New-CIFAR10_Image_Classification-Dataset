/**
 * CIFAR-10 Pure-ANN Web Application Client Script
 * Handles drag-and-drop, image preview, prediction API calls, and UI state.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const predictBtn = document.getElementById("predictBtn");
    const resetBtn = document.getElementById("resetBtn");
    const loadingState = document.getElementById("loadingState");
    const resultsContainer = document.getElementById("resultsContainer");
    const resultEmoji = document.getElementById("resultEmoji");
    const resultClass = document.getElementById("resultClass");
    const resultConfidence = document.getElementById("resultConfidence");
    const resultLatency = document.getElementById("resultLatency");
    const probBarsContainer = document.getElementById("probBarsContainer");
    const errorAlert = document.getElementById("errorAlert");
    const errorMessage = document.getElementById("errorMessage");
    const sampleChips = document.querySelectorAll(".sample-chip");
    const classPills = document.querySelectorAll(".class-pill");

    let currentFile = null;

    // -------------------------------------------------------------
    // Drag & Drop Listeners
    // -------------------------------------------------------------
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("drag-over");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    dropZone.addEventListener("click", () => {
        fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // -------------------------------------------------------------
    // Sample Chip Click Listener (Instant Test)
    // -------------------------------------------------------------
    sampleChips.forEach(chip => {
        chip.addEventListener("click", async () => {
            const className = chip.getAttribute("data-class");
            try {
                chip.style.opacity = "0.5";
                const response = await fetch(`/api/sample/${className}`);
                if (!response.ok) throw new Error("Failed to load sample image");
                const blob = await response.blob();
                const file = new File([blob], `${className}_sample.png`, { type: "image/png" });
                handleSelectedFile(file);
                chip.style.opacity = "1";
            } catch (err) {
                showError("Could not load sample: " + err.message);
                chip.style.opacity = "1";
            }
        });
    });

    // -------------------------------------------------------------
    // File Handling & Preview
    // -------------------------------------------------------------
    function handleSelectedFile(file) {
        if (!file.type.startsWith("image/")) {
            showError("Please upload a valid image file (PNG, JPG, JPEG, WEBP, BMP).");
            return;
        }

        currentFile = file;
        hideError();
        hideResults();

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove("d-none");
            dropZone.classList.add("d-none");
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // -------------------------------------------------------------
    // Predict Action (Phase 4 & 5)
    // -------------------------------------------------------------
    predictBtn.addEventListener("click", async () => {
        if (!currentFile) {
            showError("Please select or drop an image first.");
            return;
        }

        hideError();
        predictBtn.disabled = true;
        resetBtn.disabled = true;
        loadingState.classList.remove("d-none");
        resultsContainer.classList.add("d-none");

        const formData = new FormData();
        formData.append("image", currentFile);

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.message || data.error || "Prediction request failed.");
            }

            renderResults(data);
        } catch (err) {
            showError(err.message || "Failed to classify image.");
        } finally {
            loadingState.classList.add("d-none");
            predictBtn.disabled = false;
            resetBtn.disabled = false;
        }
    });

    // -------------------------------------------------------------
    // Render Results
    // -------------------------------------------------------------
    function renderResults(data) {
        resultEmoji.textContent = data.emoji || "🖼️";
        resultClass.textContent = data.predicted_class;
        resultConfidence.textContent = `${data.confidence_percentage}% Confidence`;
        resultLatency.textContent = `Inference latency: ${data.latency_ms} ms`;

        // Highlight matching class pill
        classPills.forEach(pill => {
            if (pill.getAttribute("data-class") === data.predicted_class) {
                pill.classList.add("highlight");
            } else {
                pill.classList.remove("highlight");
            }
        });

        // Render all 10 class probability progress bars
        probBarsContainer.innerHTML = "";
        const probs = data.class_probabilities || {};
        const sortedEntries = Object.entries(probs).sort((a, b) => b[1] - a[1]);

        sortedEntries.forEach(([clsName, score]) => {
            const pct = (score * 100).toFixed(1);
            const isWinner = clsName === data.predicted_class;

            const row = document.createElement("div");
            row.className = "prob-row";
            row.innerHTML = `
                <div class="prob-label ${isWinner ? 'text-cyan fw-bold' : ''}">
                    ${clsName}
                </div>
                <div class="prob-track">
                    <div class="prob-bar ${isWinner ? 'winner' : ''}" style="width: 0%"></div>
                </div>
                <div class="prob-pct ${isWinner ? 'text-cyan fw-bold' : ''}">
                    ${pct}%
                </div>
            `;
            probBarsContainer.appendChild(row);

            // Animate width
            setTimeout(() => {
                const bar = row.querySelector(".prob-bar");
                if (bar) bar.style.width = `${Math.max(pct, 1)}%`;
            }, 50);
        });

        resultsContainer.classList.remove("d-none");
        resultsContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // -------------------------------------------------------------
    // Reset Action
    // -------------------------------------------------------------
    resetBtn.addEventListener("click", () => {
        currentFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        previewContainer.classList.add("d-none");
        dropZone.classList.remove("d-none");
        predictBtn.disabled = true;
        hideResults();
        hideError();
        classPills.forEach(pill => pill.classList.remove("highlight"));
    });

    // -------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------
    function hideResults() {
        resultsContainer.classList.add("d-none");
        probBarsContainer.innerHTML = "";
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorAlert.classList.remove("d-none");
    }

    function hideError() {
        errorAlert.classList.add("d-none");
    }

    window.dismissError = hideError;
});
