import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { 
  Folder, 
  FileText, 
  ChevronRight, 
  ArrowLeft, 
  Download, 
  RefreshCw,
  FileJson,
  FileCode,
  FileSpreadsheet,
  FileImage,
  File
} from 'lucide-react';
import { useFile } from '../contexts/FileContext';

const FileBrowser = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { 
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
  } = useFile();
  
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredFiles, setFilteredFiles] = useState([]);
  const [jsonView, setJsonView] = useState('formatted'); // 'formatted' or 'raw'

  // Initialize file browser
  useEffect(() => {
    listFiles('/data');
  }, [listFiles]);

  // Filter files when search term or files change
  useEffect(() => {
    if (!files) return;
    
    if (!searchTerm) {
      setFilteredFiles(files);
      return;
    }
    
    const filtered = files.filter(file => 
      file.name.toLowerCase().includes(searchTerm.toLowerCase())
    );
    setFilteredFiles(filtered);
  }, [files, searchTerm]);

  // Handle file selection
  const handleFileSelect = (file) => {
    if (file.type === 'file') {
      getFileContent(file.path);
    } else {
      navigateTo(file.path);
    }
  };

  // Handle file download
  const handleDownloadFile = () => {
    if (!selectedFile || !fileContent) return;
    
    const blob = new Blob([fileContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = selectedFile.split('/').pop();
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Get file icon based on extension
  const getFileIcon = (filename) => {
    const ext = getFileExtension(filename);
    
    switch (ext) {
      case 'json':
        return <FileJson size={20} className="text-primary-600" />;
      case 'py':
      case 'js':
      case 'jsx':
      case 'ts':
      case 'tsx':
      case 'html':
      case 'css':
        return <FileCode size={20} className="text-primary-600" />;
      case 'csv':
      case 'xlsx':
      case 'xls':
        return <FileSpreadsheet size={20} className="text-primary-600" />;
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
      case 'svg':
        return <FileImage size={20} className="text-primary-600" />;
      case 'txt':
      case 'log':
      case 'md':
        return <FileText size={20} className="text-primary-600" />;
      default:
        return <File size={20} className="text-primary-600" />;
    }
  };

  // Render file content based on file type
  const renderFileContent = () => {
    if (!selectedFile || !fileContent) {
      return (
        <div className="h-full flex items-center justify-center text-secondary-500">
          Select a file to view its content
        </div>
      );
    }
    
    const ext = getFileExtension(selectedFile);
    
    if (ext === 'json') {
      try {
        const jsonData = JSON.parse(fileContent);
        
        if (jsonView === 'formatted') {
          return (
            <div className="overflow-auto p-4">
              <pre className="text-sm font-mono whitespace-pre-wrap">
                {JSON.stringify(jsonData, null, 2)}
              </pre>
            </div>
          );
        } else {
          return (
            <div className="overflow-auto p-4">
              <pre className="text-sm font-mono whitespace-pre-wrap">
                {fileContent}
              </pre>
            </div>
          );
        }
      } catch (e) {
        return (
          <div className="overflow-auto p-4">
            <div className="bg-error-50 border border-error-200 text-error-700 px-4 py-3 rounded mb-4">
              <p>Error parsing JSON: {e.message}</p>
            </div>
            <pre className="text-sm font-mono whitespace-pre-wrap">
              {fileContent}
            </pre>
          </div>
        );
      }
    } else if (isViewableFile(selectedFile)) {
      return (
        <div className="overflow-auto p-4">
          <pre className="text-sm font-mono whitespace-pre-wrap">
            {fileContent}
          </pre>
        </div>
      );
    } else {
      return (
        <div className="h-full flex flex-col items-center justify-center text-secondary-500">
          <p className="mb-4">This file type cannot be previewed</p>
          <button
            onClick={handleDownloadFile}
            className="btn btn-primary flex items-center"
          >
            <Download size={16} className="mr-2" />
            Download File
          </button>
        </div>
      );
    }
  };

  return (
    <div className="flex flex-col md:flex-row h-full space-y-4 md:space-y-0 md:space-x-4">
      {/* File Browser */}
      <div className="w-full md:w-1/3 bg-white rounded-lg shadow-card flex flex-col">
        {/* Path and Controls */}
        <div className="p-4 border-b border-secondary-200">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-lg font-semibold">File Browser</h2>
            <button
              onClick={() => listFiles(currentPath)}
              className="p-1 text-secondary-700 hover:text-primary-600 rounded"
              title="Refresh"
              disabled={loading}
            >
              <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
            </button>
          </div>
          <div className="flex items-center text-sm text-secondary-600 overflow-x-auto whitespace-nowrap pb-1">
            <button
              onClick={navigateUp}
              className="p-1 text-secondary-700 hover:text-primary-600 rounded mr-1"
              title="Go Up"
              disabled={currentPath === '/data'}
            >
              <ArrowLeft size={16} />
            </button>
            <span className="font-medium">Path:</span>
            {currentPath.split('/').map((segment, index, array) => {
              if (!segment) return null;
              
              const path = '/' + array.slice(1, index + 1).join('/');
              return (
                <React.Fragment key={path}>
                  <span className="mx-1">/</span>
                  <button
                    className="hover:text-primary-600"
                    onClick={() => navigateTo(path)}
                  >
                    {segment}
                  </button>
                </React.Fragment>
              );
            })}
          </div>
        </div>
        
        {/* Search */}
        <div className="px-4 py-2 border-b border-secondary-200">
          <input
            type="text"
            placeholder="Search files..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-3 py-1 text-sm border border-secondary-300 rounded-md focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-primary-500"
          />
        </div>
        
        {/* File List */}
        <div className="flex-1 overflow-auto">
          {error && (
            <div className="bg-error-50 border border-error-200 text-error-700 px-4 py-3 m-4 rounded">
              <p>{error}</p>
            </div>
          )}
          
          {loading ? (
            <div className="flex items-center justify-center h-32">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
            </div>
          ) : filteredFiles.length > 0 ? (
            <ul className="divide-y divide-secondary-100">
              {filteredFiles.map((file, index) => (
                <li key={index}>
                  <button
                    onClick={() => handleFileSelect(file)}
                    className={`w-full text-left px-4 py-2 flex items-center hover:bg-secondary-50 ${
                      selectedFile === file.path ? 'bg-primary-50' : ''
                    }`}
                  >
                    {file.type === 'directory' ? (
                      <Folder size={20} className="text-primary-600 mr-3 flex-shrink-0" />
                    ) : (
                      <span className="mr-3 flex-shrink-0">
                        {getFileIcon(file.name)}
                      </span>
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="truncate font-medium">
                        {file.name}
                      </div>
                      {file.type === 'file' && (
                        <div className="text-xs text-secondary-500">
                          {formatFileSize(file.size)}
                        </div>
                      )}
                    </div>
                    {file.type === 'directory' && (
                      <ChevronRight size={16} className="text-secondary-400" />
                    )}
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <div className="flex items-center justify-center h-32 text-secondary-500">
              No files found
            </div>
          )}
        </div>
      </div>

      {/* File Content */}
      <div className="flex-1 bg-white rounded-lg shadow-card flex flex-col">
        {/* File Header */}
        <div className="p-4 border-b border-secondary-200">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold truncate">
              {selectedFile ? selectedFile.split('/').pop() : 'File Viewer'}
            </h2>
            {selectedFile && fileContent && (
              <div className="flex items-center space-x-2">
                {getFileExtension(selectedFile) === 'json' && (
                  <div className="flex items-center space-x-1 text-sm">
                    <span>View:</span>
                    <button
                      onClick={() => setJsonView('formatted')}
                      className={`px-2 py-1 rounded ${
                        jsonView === 'formatted' 
                          ? 'bg-primary-100 text-primary-800' 
                          : 'hover:bg-secondary-100'
                      }`}
                    >
                      Formatted
                    </button>
                    <button
                      onClick={() => setJsonView('raw')}
                      className={`px-2 py-1 rounded ${
                        jsonView === 'raw' 
                          ? 'bg-primary-100 text-primary-800' 
                          : 'hover:bg-secondary-100'
                      }`}
                    >
                      Raw
                    </button>
                  </div>
                )}
                <button
                  onClick={handleDownloadFile}
                  className="p-1 text-secondary-700 hover:text-primary-600 rounded"
                  title="Download File"
                >
                  <Download size={20} />
                </button>
              </div>
            )}
          </div>
        </div>
        
        {/* File Content */}
        <div className="flex-1 overflow-hidden bg-secondary-50">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
            </div>
          ) : (
            renderFileContent()
          )}
        </div>
      </div>
    </div>
  );
};

export default FileBrowser;
