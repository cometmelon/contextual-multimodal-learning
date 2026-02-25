// background.js — Chrome Extension Background Service Worker
// Handles viewport capture (bypassing tainted canvas) and command relay.

// Listen for commands (keyboard shortcut Alt+S)
chrome.commands.onCommand.addListener((command) => {
    if (command === "toggle-selection") {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (tabs[0]) {
                chrome.tabs.sendMessage(tabs[0].id, { action: "TOGGLE_SELECTION_MODE" });
            }
        });
    }
});

// Listen for messages from the content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "CAPTURE_VIEWPORT") {
        // captureVisibleTab bypasses the tainted canvas CORS restriction
        // that prevents drawImage + toDataURL on cross-origin video elements
        chrome.tabs.captureVisibleTab(
            sender.tab.windowId,
            { format: "png", quality: 100 },
            (dataUrl) => {
                if (chrome.runtime.lastError) {
                    sendResponse({ error: chrome.runtime.lastError.message });
                } else {
                    sendResponse({ dataUrl: dataUrl });
                }
            }
        );
        // Return true to indicate we will send a response asynchronously
        return true;
    }
});
