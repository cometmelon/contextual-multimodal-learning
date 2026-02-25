// popup.js — Settings persistence for the Multimodal Video RAG extension

document.addEventListener("DOMContentLoaded", () => {
    const backendUrlInput = document.getElementById("backend-url");
    const apiKeyInput = document.getElementById("api-key");
    const saveBtn = document.getElementById("save-btn");

    // Load saved settings
    chrome.storage.local.get(["backendUrl", "apiKey"], (result) => {
        if (result.backendUrl) backendUrlInput.value = result.backendUrl;
        if (result.apiKey) apiKeyInput.value = result.apiKey;
    });

    // Save settings
    saveBtn.addEventListener("click", () => {
        const settings = {
            backendUrl: backendUrlInput.value.trim() || "http://localhost:8000",
            apiKey: apiKeyInput.value.trim(),
        };

        chrome.storage.local.set(settings, () => {
            saveBtn.textContent = "Saved ✓";
            saveBtn.classList.add("saved");

            setTimeout(() => {
                saveBtn.textContent = "Save Settings";
                saveBtn.classList.remove("saved");
            }, 2000);
        });
    });
});
