import React, { createContext, useContext, useState, useCallback } from 'react';
import { fileService } from '../services/fileService';

const FileContext = createContext();

export const useFile = () => {
  const context = useContext(FileContext);
  if (!context) {
    throw new Error('useFile must be used within a FileProvider');
  }
  return context;
};

export const FileProvider = ({ children }) => {
  const [currentPath, setCurrentPath] = useState('/data');
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileContent, setFileContent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // List files in a directory
  const listFiles = useCallback(async (path = currentPath) => {
    setLoading(true);
    try {
      const data = await fileService.listFiles(path);
      setFiles(data);
      setCurrentPath(path);
      setError(null);
      return data;
    } catch (err) {
      console.error('Error listing files:', err);
      setError(`Failed to list files: ${err.message}`);
      return [];
    } finally {
      setLoading(false);
    }
  }, [currentPath]);

  // Get file content
  const getFileContent = useCallback(async (path) => {
    setLoading(true);
    try {
      const content = await fileService.getFileContent(path);
      setFileContent(content);
      setSelectedFile(path);
      setError(null);
      return content;
    } catch (err) {
      console.error('Error getting file content:', err);
      setError(`Failed to get file content: ${err.message}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Navigate to parent directory
  const navigateUp = useCallback(() => {
    const parentPath = currentPath.split('/').slice(0, -1).join('/') || '/';
    listFiles(parentPath);
  }, [currentPath, listFiles]);

  // Navigate to a directory
  const navigateTo = useCallback((path) => {
    listFiles(path);
  }, [listFiles]);

  // Select a file
  const selectFile = useCallback((file) => {
    if (file.type === 'file') {
      getFileContent(file.path);
    } else {
      navigateTo(file.path);
    }
  }, [getFileContent, navigateTo]);

  // Get file extension
  const getFileExtension = useCallback((filename) => {
    return filename.split('.').pop().toLowerCase();
  }, []);

  // Check if file is viewable
  const isViewableFile = useCallback((filename) => {
    const ext = getFileExtension(filename);
    return ['json', 'txt', 'log', 'py', 'js', 'html', 'css', 'md'].includes(ext);
  }, [getFileExtension]);

  // Format file size
  const formatFileSize = useCallback((size) => {
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    if (size < 1024 * 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
    return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }, []);

  const value = {
    currentPath,
    files,
    selectedFile,
    fileContent,
    loading,
    error,
    listFiles,
    getFileContent,
    navigateUp,
    navigateTo,
    selectFile,
    getFileExtension,
    isViewableFile,
    formatFileSize
  };

  return (
    <FileContext.Provider value={value}>
      {children}
    </FileContext.Provider>
  );
};
