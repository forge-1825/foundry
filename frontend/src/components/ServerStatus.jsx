import React, { useEffect, useState } from 'react';
import { Alert, AlertTitle, AlertDescription } from './ui/alert';

export function ServerStatus() {
  const [serverStatus, setServerStatus] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await fetch('/api/server-status');
        const data = await response.json();
        setServerStatus(data);
      } catch (error) {
        console.error('Failed to check server status:', error);
      } finally {
        setLoading(false);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <Alert variant="info">Checking server status...</Alert>;
  }

  return (
    <div className="space-y-4">
      {Object.entries(serverStatus).map(([type, status]) => (
        <Alert 
          key={type}
          variant={status.running ? 'success' : 'destructive'}
        >
          <AlertTitle>{status.name}</AlertTitle>
          <AlertDescription>
            {status.running ? (
              `Running on port ${status.port}`
            ) : (
              <>
                Not running. Please start the container:
                <code className="block mt-2 p-2 bg-gray-100 rounded">
                  docker start {status.container}
                </code>
              </>
            )}
          </AlertDescription>
        </Alert>
      ))}
    </div>
  );
}


