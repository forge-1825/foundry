import React, { useState, useEffect } from 'react';
import { AlertCircle, CheckCircle, XCircle, Server, Loader } from 'lucide-react';
import { scriptService } from '../services/scriptService';

const ModelAvailabilityStatus = ({ onRefresh = null }) => {
  const [loading, setLoading] = useState(true);
  const [availability, setAvailability] = useState(null);
  const [error, setError] = useState(null);

  const checkAvailability = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await scriptService.checkModelAvailability();
      setAvailability(data);
    } catch (err) {
      console.error('Error checking model availability:', err);
      setError('Failed to check model availability');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkAvailability();
  }, []);

  const getStatusColor = () => {
    if (loading) return 'text-gray-500';
    if (error) return 'text-red-500';
    if (!availability) return 'text-gray-500';
    
    const activeServers = availability.summary.active_servers;
    if (activeServers === 0) return 'text-red-500';
    if (activeServers === 1) return 'text-yellow-500';
    return 'text-green-500';
  };

  const getStatusIcon = () => {
    if (loading) return <Loader className="animate-spin" size={20} />;
    if (error) return <XCircle size={20} />;
    if (!availability) return <AlertCircle size={20} />;
    
    const activeServers = availability.summary.active_servers;
    if (activeServers === 0) return <XCircle size={20} />;
    if (activeServers === 1) return <AlertCircle size={20} />;
    return <CheckCircle size={20} />;
  };

  const getStatusMessage = () => {
    if (loading) return 'Checking model availability...';
    if (error) return error;
    if (!availability) return 'Unable to check model availability';
    
    const { active_servers, total_servers } = availability.summary;
    if (active_servers === 0) {
      return 'No models available - Please start Docker containers';
    }
    return `${active_servers} of ${total_servers} models active`;
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-700 flex items-center">
          <Server size={16} className="mr-2" />
          Model Server Status
        </h3>
        <button
          onClick={() => {
            checkAvailability();
            if (onRefresh) onRefresh();
          }}
          className="text-sm text-blue-600 hover:text-blue-800"
          disabled={loading}
        >
          Refresh
        </button>
      </div>

      <div className={`flex items-center ${getStatusColor()}`}>
        {getStatusIcon()}
        <span className="ml-2 text-sm">{getStatusMessage()}</span>
      </div>

      {availability && availability.servers.length > 0 && (
        <div className="mt-3 space-y-2">
          <div className="text-xs text-gray-600 font-medium">Available Servers:</div>
          {availability.servers.map((server, index) => (
            <div
              key={index}
              className="flex items-center justify-between text-xs bg-gray-50 rounded px-2 py-1"
            >
              <span className="flex items-center">
                <span
                  className={`w-2 h-2 rounded-full mr-2 ${
                    server.status === 'active' ? 'bg-green-500' : 'bg-gray-400'
                  }`}
                />
                {server.name}
              </span>
              <span className="text-gray-500">
                Port {server.port} ({server.type})
              </span>
            </div>
          ))}
        </div>
      )}

      {availability && availability.recommendations.length > 0 && (
        <div className="mt-3 space-y-2">
          {availability.recommendations.map((rec, index) => (
            <div
              key={index}
              className={`text-xs p-2 rounded ${
                rec.type === 'error'
                  ? 'bg-red-50 text-red-700'
                  : rec.type === 'warning'
                  ? 'bg-yellow-50 text-yellow-700'
                  : 'bg-blue-50 text-blue-700'
              }`}
            >
              {rec.message}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ModelAvailabilityStatus;