import api from './apiService';

// Since the backend doesn't have dedicated file endpoints yet,
// we'll create a mock implementation that we can replace later
export const fileService = {
  // List files in a directory
  listFiles: async (path) => {
    // This is a placeholder - in a real implementation, we would call an API endpoint
    // return api.get(`/api/files?path=${encodeURIComponent(path)}`);
    
    // Mock implementation
    return new Promise((resolve) => {
      setTimeout(() => {
        // Generate some mock files based on the path
        const files = [];
        
        // Add parent directory if not at root
        if (path !== '/' && path !== '/data') {
          files.push({
            name: '..',
            path: path.split('/').slice(0, -1).join('/') || '/',
            type: 'directory',
            size: 0,
            modified: new Date().toISOString()
          });
        }
        
        // Add some mock directories
        if (path === '/data') {
          files.push({
            name: 'extracted',
            path: '/data/extracted',
            type: 'directory',
            size: 0,
            modified: new Date().toISOString()
          });
          files.push({
            name: 'enriched',
            path: '/data/enriched',
            type: 'directory',
            size: 0,
            modified: new Date().toISOString()
          });
          files.push({
            name: 'teacher_pairs',
            path: '/data/teacher_pairs',
            type: 'directory',
            size: 0,
            modified: new Date().toISOString()
          });
          files.push({
            name: 'distilled_model',
            path: '/data/distilled_model',
            type: 'directory',
            size: 0,
            modified: new Date().toISOString()
          });
          files.push({
            name: 'configs',
            path: '/data/configs',
            type: 'directory',
            size: 0,
            modified: new Date().toISOString()
          });
          files.push({
            name: 'history',
            path: '/data/history',
            type: 'directory',
            size: 0,
            modified: new Date().toISOString()
          });
        } else if (path === '/data/extracted') {
          files.push({
            name: 'extracted_data.json',
            path: '/data/extracted/extracted_data.json',
            type: 'file',
            size: 1024 * 1024 * 2, // 2MB
            modified: new Date().toISOString()
          });
          files.push({
            name: 'pdfs',
            path: '/data/extracted/pdfs',
            type: 'directory',
            size: 0,
            modified: new Date().toISOString()
          });
        } else if (path === '/data/enriched') {
          files.push({
            name: 'enriched_data.json',
            path: '/data/enriched/enriched_data.json',
            type: 'file',
            size: 1024 * 1024 * 3, // 3MB
            modified: new Date().toISOString()
          });
        } else if (path === '/data/teacher_pairs') {
          files.push({
            name: 'teacher_pairs.json',
            path: '/data/teacher_pairs/teacher_pairs.json',
            type: 'file',
            size: 1024 * 1024 * 5, // 5MB
            modified: new Date().toISOString()
          });
        } else if (path === '/data/distilled_model') {
          files.push({
            name: 'training_metrics.json',
            path: '/data/distilled_model/training_metrics.json',
            type: 'file',
            size: 1024 * 10, // 10KB
            modified: new Date().toISOString()
          });
          files.push({
            name: 'best_checkpoint',
            path: '/data/distilled_model/best_checkpoint',
            type: 'directory',
            size: 0,
            modified: new Date().toISOString()
          });
          files.push({
            name: 'logs',
            path: '/data/distilled_model/logs',
            type: 'directory',
            size: 0,
            modified: new Date().toISOString()
          });
        } else if (path === '/data/configs') {
          files.push({
            name: 'manual_extractor_config.json',
            path: '/data/configs/manual_extractor_config.json',
            type: 'file',
            size: 1024, // 1KB
            modified: new Date().toISOString()
          });
          files.push({
            name: 'data_enrichment_config.json',
            path: '/data/configs/data_enrichment_config.json',
            type: 'file',
            size: 1024, // 1KB
            modified: new Date().toISOString()
          });
          files.push({
            name: 'teacher_pair_generation_config.json',
            path: '/data/configs/teacher_pair_generation_config.json',
            type: 'file',
            size: 1024, // 1KB
            modified: new Date().toISOString()
          });
          files.push({
            name: 'distillation_config.json',
            path: '/data/configs/distillation_config.json',
            type: 'file',
            size: 1024, // 1KB
            modified: new Date().toISOString()
          });
        } else if (path === '/data/history') {
          files.push({
            name: 'run_history.json',
            path: '/data/history/run_history.json',
            type: 'file',
            size: 1024 * 50, // 50KB
            modified: new Date().toISOString()
          });
        }
        
        resolve(files);
      }, 500); // Simulate network delay
    });
  },

  // Get file content
  getFileContent: async (path) => {
    // This is a placeholder - in a real implementation, we would call an API endpoint
    // return api.get(`/api/files/content?path=${encodeURIComponent(path)}`);
    
    // Mock implementation
    return new Promise((resolve) => {
      setTimeout(() => {
        // Generate some mock content based on the file path
        let content = '';
        
        if (path.endsWith('.json')) {
          if (path.includes('extracted_data')) {
            content = JSON.stringify([
              {
                "url": "https://example.com/doc1",
                "title": "Example Document 1",
                "publication_date": "2023-01-15",
                "section_headers": ["Introduction", "Methods", "Results", "Conclusion"],
                "raw_content": "This is an example document with some content..."
              },
              {
                "url": "https://example.com/doc2",
                "title": "Example Document 2",
                "publication_date": "2023-02-20",
                "section_headers": ["Abstract", "Introduction", "Methods", "Discussion"],
                "raw_content": "Another example document with different content..."
              }
            ], null, 2);
          } else if (path.includes('enriched_data')) {
            content = JSON.stringify([
              {
                "url": "https://example.com/doc1",
                "cleaned_text": "example document content methods results conclusion",
                "entities": [
                  {"text": "Methods", "label": "SECTION"},
                  {"text": "Results", "label": "SECTION"}
                ],
                "summary": "This is a summary of the example document."
              },
              {
                "url": "https://example.com/doc2",
                "cleaned_text": "another example document abstract introduction methods discussion",
                "entities": [
                  {"text": "Abstract", "label": "SECTION"},
                  {"text": "Introduction", "label": "SECTION"}
                ],
                "summary": "This is a summary of another example document."
              }
            ], null, 2);
          } else if (path.includes('teacher_pairs')) {
            content = JSON.stringify([
              {
                "url": "https://example.com/doc1",
                "input": "This is a summary of the example document.",
                "target": "The document describes an example with methods and results sections."
              },
              {
                "url": "https://example.com/doc2",
                "input": "This is a summary of another example document.",
                "target": "The document contains an abstract, introduction, methods, and discussion."
              }
            ], null, 2);
          } else if (path.includes('training_metrics')) {
            content = JSON.stringify({
              "epochs": 5,
              "steps": 100,
              "final_loss": 0.1234,
              "validation_loss": 0.2345,
              "training_time": "00:45:30"
            }, null, 2);
          } else if (path.includes('config')) {
            content = JSON.stringify({
              "name": path.split('/').pop().replace('_config.json', ''),
              "parameters": {
                "param1": "value1",
                "param2": 123,
                "param3": true
              },
              "created": new Date().toISOString()
            }, null, 2);
          } else if (path.includes('run_history')) {
            content = JSON.stringify([
              {
                "scriptId": "manual_extractor",
                "timestamp": "2023-10-15T14:30:00Z",
                "duration": 120,
                "status": "completed",
                "config": {
                  "url": "https://example.com/docs",
                  "output_dir": "/data/extracted"
                }
              },
              {
                "scriptId": "data_enrichment",
                "timestamp": "2023-10-15T14:35:00Z",
                "duration": 180,
                "status": "completed",
                "config": {
                  "input_file": "/data/extracted/extracted_data.json",
                  "output_file": "/data/enriched/enriched_data.json"
                }
              }
            ], null, 2);
          }
        } else if (path.endsWith('.log')) {
          content = `[2023-10-15 14:30:00] [INFO] Starting script execution
[2023-10-15 14:30:01] [INFO] Loading data from /data/extracted/extracted_data.json
[2023-10-15 14:30:02] [INFO] Processing 10 records
[2023-10-15 14:30:10] [WARNING] Record 3 has missing fields
[2023-10-15 14:30:20] [INFO] Completed processing 9/10 records successfully
[2023-10-15 14:30:21] [ERROR] Failed to process record 10: Invalid format
[2023-10-15 14:30:22] [INFO] Saving results to /data/enriched/enriched_data.json
[2023-10-15 14:30:23] [INFO] Script execution completed`;
        } else {
          content = `File content not available for ${path}`;
        }
        
        resolve(content);
      }, 500); // Simulate network delay
    });
  }
};
