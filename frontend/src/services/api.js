// API configuration
export const API_BASE_URL = window._env_?.REACT_APP_API_URL || process.env.REACT_APP_API_URL || "http://localhost:7433";

// API endpoints
export const API_ENDPOINTS = {
    // Novelty insights endpoints
    NOVELTY_DATA: "/api/novelty/data",
    NOVELTY_LOGS: "/api/novelty/logs",
    NOVELTY_TIMELINE: "/api/novelty/timeline",
    NOVELTY_MOST_NOVEL: "/api/novelty/most-novel",
    NOVELTY_LEAST_NOVEL: "/api/novelty/least-novel",
    NOVELTY_STATISTICS: "/api/novelty/statistics",
    
    // System endpoints
    SYSTEM_STATUS: "/api/system/status",
    
    // Script endpoints
    SCRIPTS: "/api/scripts",
    SCRIPT_STATUS: (scriptId) => `/api/scripts/${scriptId}/status`,
    SCRIPT_LOGS: (scriptId) => `/api/scripts/${scriptId}/logs`,
    SCRIPT_EXECUTE: (scriptId) => `/api/scripts/${scriptId}/execute`,
    SCRIPT_CONFIG: (scriptId) => `/api/scripts/${scriptId}/config`,
    
    // Watchdog endpoints
    WATCHDOG_STATUS: "/api/watchdog/status",
    WATCHDOG_ACTION: "/api/watchdog/action",
};

// WebSocket endpoints
export const WS_ENDPOINTS = {
    SCRIPT_LOGS: (scriptId) => `/ws/scripts/${scriptId}/logs`,
    STATUS: "/ws/status",
    RESOURCES: "/ws/resources",
};

// Helper function to build full API URL
export const buildApiUrl = (endpoint) => `${API_BASE_URL}${endpoint}`;

// Helper function to build full WebSocket URL
export const buildWsUrl = (endpoint) => {
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const baseUrl = API_BASE_URL.replace(/^https?:\/\//, "");
    return `${wsProtocol}//${baseUrl}${endpoint}`;
};
