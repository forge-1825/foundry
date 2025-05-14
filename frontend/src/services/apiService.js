import axios from 'axios';

// Create an axios instance with default config
const api = axios.create({
  baseURL: window._env_?.REACT_APP_API_URL || process.env.REACT_APP_API_URL || '',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add a request interceptor
api.interceptors.request.use(
  (config) => {
    // You can add auth tokens or other headers here
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add a response interceptor
api.interceptors.response.use(
  (response) => {
    // Any status code within the range of 2xx
    return response.data;
  },
  (error) => {
    // Any status codes outside the range of 2xx
    console.error('API Error:', error.response || error);
    
    // Enhance error with more details
    const enhancedError = new Error(
      error.response?.data?.detail || error.message || 'Unknown error'
    );
    enhancedError.status = error.response?.status;
    enhancedError.data = error.response?.data;
    
    return Promise.reject(enhancedError);
  }
);

export default api;
