import { useState, useEffect, useCallback, useRef } from 'react';

export const useWebSocket = (url) => {
  const [lastMessage, setLastMessage] = useState(null);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const urlRef = useRef(url);

  // Update the URL ref when the URL changes
  useEffect(() => {
    urlRef.current = url;
  }, [url]);

  // Function to connect to the WebSocket
  const connect = useCallback(() => {
    // Clear any existing reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Don't connect if no URL is provided
    if (!urlRef.current) {
      setConnected(false);
      return;
    }

    // Create the WebSocket connection
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}${urlRef.current}`;
    const socket = new WebSocket(wsUrl);
    socketRef.current = socket;

    // Set up event handlers
    socket.onopen = () => {
      console.log(`WebSocket connected: ${wsUrl}`);
      setConnected(true);
    };

    socket.onmessage = (event) => {
      setLastMessage(event.data);
    };

    socket.onclose = (event) => {
      console.log(`WebSocket closed: ${event.code} ${event.reason}`);
      setConnected(false);

      // Attempt to reconnect after a delay
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log('Attempting to reconnect WebSocket...');
        connect();
      }, 3000);
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    // Clean up function
    return () => {
      if (socket) {
        socket.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  // Connect to the WebSocket when the component mounts or the URL changes
  useEffect(() => {
    const cleanup = connect();
    
    // Clean up when the component unmounts
    return () => {
      if (cleanup) cleanup();
    };
  }, [connect, url]);

  // Function to send a message through the WebSocket
  const sendMessage = useCallback((message) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(typeof message === 'string' ? message : JSON.stringify(message));
      return true;
    }
    return false;
  }, []);

  return { lastMessage, sendMessage, connected };
};
