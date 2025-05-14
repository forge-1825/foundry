import React, { useState, useEffect } from 'react';
import { Card, Form, Button, Table, Tabs, Tab, Alert } from 'react-bootstrap';
import { Bar, Line, Radar, Pie } from 'react-chartjs-2';
import { Chart as ChartJS, registerables } from 'chart.js';

// Register all ChartJS components
ChartJS.register(...registerables);

const ResultsVisualization = ({ resultDetails, visualizationConfig, onConfigChange }) => {
  const [chartData, setChartData] = useState(null);
  const [availableMetrics, setAvailableMetrics] = useState([]);
  const [availableModels, setAvailableModels] = useState([]);
  const [availableEvaluators, setAvailableEvaluators] = useState([]);

  // Extract available metrics, models, and evaluators from result details
  useEffect(() => {
    if (!resultDetails) return;

    // Extract models
    const models = Object.keys(resultDetails.models || {}).map(modelId => ({
      id: modelId,
      name: resultDetails.models[modelId].name || modelId
    }));
    setAvailableModels(models);

    // Extract evaluators
    const evaluators = Object.keys(resultDetails.evaluator_scores || {}).map(evaluatorId => ({
      id: evaluatorId,
      name: evaluatorId
    }));
    setAvailableEvaluators(evaluators);

    // Extract metrics from the first model's scores
    if (models.length > 0 && resultDetails.models[models[0].id]?.scores) {
      const firstModelScores = resultDetails.models[models[0].id].scores;
      const metrics = Object.keys(firstModelScores).map(metricId => ({
        id: metricId,
        name: metricId
      }));
      setAvailableMetrics(metrics);
    }

    // Generate chart data based on current configuration
    generateChartData();
  }, [resultDetails, visualizationConfig]);

  // Generate chart data based on configuration
  const generateChartData = () => {
    if (!resultDetails) return;

    const { chartType, compareBy, metrics } = visualizationConfig;
    
    if (compareBy === 'model') {
      // Compare models across selected metrics
      const labels = Object.keys(resultDetails.models || {}).map(modelId => 
        resultDetails.models[modelId].name || modelId
      );
      
      const datasets = metrics.map(metricId => {
        const data = Object.keys(resultDetails.models || {}).map(modelId => {
          const modelScores = resultDetails.models[modelId].scores || {};
          return modelScores[metricId]?.average || 0;
        });
        
        return {
          label: metricId,
          data,
          backgroundColor: getRandomColor(metricId),
          borderColor: getRandomColor(metricId, 0.8),
          borderWidth: 1
        };
      });
      
      setChartData({ labels, datasets });
    } else if (compareBy === 'metric') {
      // Compare metrics across selected models
      const labels = metrics;
      
      const datasets = Object.keys(resultDetails.models || {})
        .filter(modelId => visualizationConfig.selectedModels?.includes(modelId))
        .map(modelId => {
          const modelScores = resultDetails.models[modelId].scores || {};
          const data = metrics.map(metricId => modelScores[metricId]?.average || 0);
          
          return {
            label: resultDetails.models[modelId].name || modelId,
            data,
            backgroundColor: getRandomColor(modelId),
            borderColor: getRandomColor(modelId, 0.8),
            borderWidth: 1
          };
        });
      
      setChartData({ labels, datasets });
    }
  };

  // Generate a random color based on a string
  const getRandomColor = (str, alpha = 0.5) => {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    
    const r = (hash & 0xFF) % 256;
    const g = ((hash >> 8) & 0xFF) % 256;
    const b = ((hash >> 16) & 0xFF) % 256;
    
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  // Handle configuration changes
  const handleConfigChange = (field, value) => {
    const newConfig = {
      ...visualizationConfig,
      [field]: value
    };
    onConfigChange(newConfig);
  };

  // Handle metric selection
  const handleMetricSelect = (metricId) => {
    const newMetrics = [...visualizationConfig.metrics];
    const index = newMetrics.indexOf(metricId);
    
    if (index === -1) {
      newMetrics.push(metricId);
    } else {
      newMetrics.splice(index, 1);
    }
    
    handleConfigChange('metrics', newMetrics);
  };

  // Handle model selection
  const handleModelSelect = (modelId) => {
    const newSelectedModels = [...(visualizationConfig.selectedModels || [])];
    const index = newSelectedModels.indexOf(modelId);
    
    if (index === -1) {
      newSelectedModels.push(modelId);
    } else {
      newSelectedModels.splice(index, 1);
    }
    
    handleConfigChange('selectedModels', newSelectedModels);
  };

  // Render chart based on type
  const renderChart = () => {
    if (!chartData) return <Alert variant="info">No data available for visualization</Alert>;
    
    const options = {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 1.0
        }
      }
    };
    
    switch (visualizationConfig.chartType) {
      case 'bar':
        return <Bar data={chartData} options={options} height={300} />;
      case 'line':
        return <Line data={chartData} options={options} height={300} />;
      case 'radar':
        return <Radar data={chartData} options={{ ...options, scales: undefined }} height={300} />;
      case 'pie':
        return <Pie data={chartData} options={{ ...options, scales: undefined }} height={300} />;
      default:
        return <Bar data={chartData} options={options} height={300} />;
    }
  };

  // Render raw data table
  const renderRawDataTable = () => {
    if (!resultDetails) return null;
    
    return (
      <div className="mt-4">
        <h5>Raw Evaluation Scores</h5>
        <Table striped bordered hover size="sm">
          <thead>
            <tr>
              <th>Model</th>
              {availableMetrics.map(metric => (
                <th key={metric.id}>{metric.name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {availableModels.map(model => (
              <tr key={model.id}>
                <td>{model.name}</td>
                {availableMetrics.map(metric => {
                  const score = resultDetails.models[model.id]?.scores?.[metric.id]?.average || 0;
                  return (
                    <td key={`${model.id}-${metric.id}`}>
                      {score.toFixed(4)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </Table>
      </div>
    );
  };

  if (!resultDetails) {
    return <Alert variant="info">Select an evaluation result to visualize</Alert>;
  }

  return (
    <Card>
      <Card.Header>
        <h4>Results Visualization</h4>
      </Card.Header>
      <Card.Body>
        <Tabs defaultActiveKey="chart" className="mb-3">
          <Tab eventKey="chart" title="Chart">
            <div className="mb-3">
              <Form.Group className="mb-3">
                <Form.Label>Chart Type</Form.Label>
                <Form.Select
                  value={visualizationConfig.chartType}
                  onChange={(e) => handleConfigChange('chartType', e.target.value)}
                >
                  <option value="bar">Bar Chart</option>
                  <option value="line">Line Chart</option>
                  <option value="radar">Radar Chart</option>
                  <option value="pie">Pie Chart</option>
                </Form.Select>
              </Form.Group>
              
              <Form.Group className="mb-3">
                <Form.Label>Compare By</Form.Label>
                <Form.Select
                  value={visualizationConfig.compareBy}
                  onChange={(e) => handleConfigChange('compareBy', e.target.value)}
                >
                  <option value="model">Compare Models</option>
                  <option value="metric">Compare Metrics</option>
                </Form.Select>
              </Form.Group>
              
              <Form.Group className="mb-3">
                <Form.Label>Select Metrics</Form.Label>
                <div className="d-flex flex-wrap gap-2">
                  {availableMetrics.map(metric => (
                    <Form.Check
                      key={metric.id}
                      type="checkbox"
                      id={`metric-${metric.id}`}
                      label={metric.name}
                      checked={visualizationConfig.metrics.includes(metric.id)}
                      onChange={() => handleMetricSelect(metric.id)}
                      inline
                    />
                  ))}
                </div>
              </Form.Group>
              
              {visualizationConfig.compareBy === 'metric' && (
                <Form.Group className="mb-3">
                  <Form.Label>Select Models</Form.Label>
                  <div className="d-flex flex-wrap gap-2">
                    {availableModels.map(model => (
                      <Form.Check
                        key={model.id}
                        type="checkbox"
                        id={`model-${model.id}`}
                        label={model.name}
                        checked={(visualizationConfig.selectedModels || []).includes(model.id)}
                        onChange={() => handleModelSelect(model.id)}
                        inline
                      />
                    ))}
                  </div>
                </Form.Group>
              )}
              
              <Form.Group className="mb-3">
                <Form.Check
                  type="checkbox"
                  id="show-raw-data"
                  label="Show Raw Data Table"
                  checked={visualizationConfig.showRawData}
                  onChange={(e) => handleConfigChange('showRawData', e.target.checked)}
                />
              </Form.Group>
            </div>
            
            <div style={{ height: '400px' }}>
              {renderChart()}
            </div>
            
            {visualizationConfig.showRawData && renderRawDataTable()}
          </Tab>
          
          <Tab eventKey="examples" title="Example Results">
            <div className="mt-3">
              <h5>Example Results</h5>
              {availableModels.map(model => {
                const examples = resultDetails.models[model.id]?.examples || [];
                if (examples.length === 0) return null;
                
                return (
                  <div key={model.id} className="mb-4">
                    <h6>{model.name} - First 3 Examples</h6>
                    <Table striped bordered hover size="sm">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Input</th>
                          <th>Output</th>
                          <th>Reference</th>
                          <th>Scores</th>
                        </tr>
                      </thead>
                      <tbody>
                        {examples.slice(0, 3).map((example, index) => (
                          <tr key={index}>
                            <td>{index + 1}</td>
                            <td>
                              <div style={{ maxHeight: '100px', overflow: 'auto' }}>
                                {typeof example.input === 'string'
                                  ? example.input
                                  : JSON.stringify(example.input, null, 2)}
                              </div>
                            </td>
                            <td>
                              <div style={{ maxHeight: '100px', overflow: 'auto' }}>
                                {example.output}
                              </div>
                            </td>
                            <td>
                              <div style={{ maxHeight: '100px', overflow: 'auto' }}>
                                {example.reference_output}
                              </div>
                            </td>
                            <td>
                              <div style={{ maxHeight: '100px', overflow: 'auto' }}>
                                {Object.entries(example.scores || {}).map(([scoreId, scoreData]) => (
                                  <div key={scoreId}>
                                    <strong>{scoreId}:</strong> {scoreData.score?.toFixed(4) || 'N/A'}
                                  </div>
                                ))}
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </div>
                );
              })}
            </div>
          </Tab>
        </Tabs>
      </Card.Body>
    </Card>
  );
};

export default ResultsVisualization;
