import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Form, Tabs, Tab, Alert, Table, Badge, Spinner, Dropdown } from 'react-bootstrap';
import { useParams, useNavigate } from 'react-router-dom';
import { Bar, Line, Radar, Pie } from 'react-chartjs-2';
import { Chart as ChartJS, registerables } from 'chart.js';
import axios from 'axios';

// Register all ChartJS components
ChartJS.register(...registerables);

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:7433';

const EvaluationResultsVisualization = () => {
  const { runId } = useParams();
  const navigate = useNavigate();
  
  // State for evaluation results
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [resultDetails, setResultDetails] = useState(null);
  
  // State for visualization configuration
  const [visualizationConfig, setVisualizationConfig] = useState({
    chartType: 'bar',
    compareBy: 'model',
    metrics: [],
    selectedModels: [],
    selectedEvaluators: [],
    showRawData: true,
    sortBy: 'name',
    sortDirection: 'asc',
    filterText: ''
  });
  
  // State for available options
  const [availableMetrics, setAvailableMetrics] = useState([]);
  const [availableModels, setAvailableModels] = useState([]);
  const [availableEvaluators, setAvailableEvaluators] = useState([]);
  
  // State for chart data
  const [chartData, setChartData] = useState(null);
  
  // Fetch evaluation result details
  useEffect(() => {
    const fetchResultDetails = async () => {
      if (!runId) return;
      
      setLoading(true);
      setError(null);
      
      try {
        const response = await axios.get(`${API_URL}/api/evaluation/results/${runId}`);
        setResultDetails(response.data);
        
        // Extract available metrics, models, and evaluators
        const models = Object.keys(response.data.models || {}).map(modelId => ({
          id: modelId,
          name: response.data.models[modelId].name || modelId,
          type: response.data.models[modelId].type || 'unknown'
        }));
        setAvailableModels(models);
        
        // Set default selected models (all)
        setVisualizationConfig(prev => ({
          ...prev,
          selectedModels: models.map(m => m.id)
        }));
        
        // Extract evaluators
        const evaluators = Object.keys(response.data.evaluator_scores || {}).map(evaluatorId => ({
          id: evaluatorId,
          name: evaluatorId
        }));
        setAvailableEvaluators(evaluators);
        
        // Set default selected evaluators (all)
        setVisualizationConfig(prev => ({
          ...prev,
          selectedEvaluators: evaluators.map(e => e.id)
        }));
        
        // Extract metrics from the first model's scores
        if (models.length > 0 && response.data.models[models[0].id]?.scores) {
          const firstModelScores = response.data.models[models[0].id].scores;
          const metrics = Object.keys(firstModelScores).map(metricId => ({
            id: metricId,
            name: metricId
          }));
          setAvailableMetrics(metrics);
          
          // Set default selected metrics (first 3 or all if less than 3)
          setVisualizationConfig(prev => ({
            ...prev,
            metrics: metrics.slice(0, Math.min(3, metrics.length)).map(m => m.id)
          }));
        }
      } catch (err) {
        console.error('Error fetching result details:', err);
        setError('Failed to fetch evaluation result details. Please try again.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchResultDetails();
  }, [runId]);
  
  // Generate chart data when configuration or result details change
  useEffect(() => {
    if (!resultDetails) return;
    
    generateChartData();
  }, [resultDetails, visualizationConfig]);
  
  // Generate chart data based on configuration
  const generateChartData = () => {
    const { chartType, compareBy, metrics, selectedModels, selectedEvaluators } = visualizationConfig;
    
    if (compareBy === 'model') {
      // Compare models across selected metrics
      // Only include selected models
      const filteredModels = availableModels
        .filter(model => selectedModels.includes(model.id))
        .sort((a, b) => {
          if (visualizationConfig.sortDirection === 'asc') {
            return a.name.localeCompare(b.name);
          } else {
            return b.name.localeCompare(a.name);
          }
        });
      
      const labels = filteredModels.map(model => model.name);
      
      const datasets = metrics.map(metricId => {
        const data = filteredModels.map(model => {
          const modelScores = resultDetails.models[model.id].scores || {};
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
      
      const datasets = availableModels
        .filter(model => selectedModels.includes(model.id))
        .sort((a, b) => {
          if (visualizationConfig.sortDirection === 'asc') {
            return a.name.localeCompare(b.name);
          } else {
            return b.name.localeCompare(a.name);
          }
        })
        .map(model => {
          const modelScores = resultDetails.models[model.id].scores || {};
          const data = metrics.map(metricId => modelScores[metricId]?.average || 0);
          
          return {
            label: model.name,
            data,
            backgroundColor: getRandomColor(model.id),
            borderColor: getRandomColor(model.id, 0.8),
            borderWidth: 1
          };
        });
      
      setChartData({ labels, datasets });
    } else if (compareBy === 'evaluator') {
      // Compare evaluators across selected models
      const labels = selectedEvaluators;
      
      const datasets = availableModels
        .filter(model => selectedModels.includes(model.id))
        .sort((a, b) => {
          if (visualizationConfig.sortDirection === 'asc') {
            return a.name.localeCompare(b.name);
          } else {
            return b.name.localeCompare(a.name);
          }
        })
        .map(model => {
          const data = selectedEvaluators.map(evaluatorId => {
            const evaluatorScores = resultDetails.evaluator_scores[evaluatorId] || {};
            return evaluatorScores[model.id] || 0;
          });
          
          return {
            label: model.name,
            data,
            backgroundColor: getRandomColor(model.id),
            borderColor: getRandomColor(model.id, 0.8),
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
  
  // Format date string
  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown';
    
    const date = new Date(dateString);
    return date.toLocaleString();
  };
  
  // Handle configuration changes
  const handleConfigChange = (field, value) => {
    setVisualizationConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  // Handle metric selection
  const handleMetricSelect = (metricId) => {
    setVisualizationConfig(prev => {
      const metrics = [...prev.metrics];
      const index = metrics.indexOf(metricId);
      
      if (index === -1) {
        metrics.push(metricId);
      } else {
        metrics.splice(index, 1);
      }
      
      return {
        ...prev,
        metrics
      };
    });
  };
  
  // Handle model selection
  const handleModelSelect = (modelId) => {
    setVisualizationConfig(prev => {
      const selectedModels = [...prev.selectedModels];
      const index = selectedModels.indexOf(modelId);
      
      if (index === -1) {
        selectedModels.push(modelId);
      } else {
        selectedModels.splice(index, 1);
      }
      
      return {
        ...prev,
        selectedModels
      };
    });
  };
  
  // Handle evaluator selection
  const handleEvaluatorSelect = (evaluatorId) => {
    setVisualizationConfig(prev => {
      const selectedEvaluators = [...prev.selectedEvaluators];
      const index = selectedEvaluators.indexOf(evaluatorId);
      
      if (index === -1) {
        selectedEvaluators.push(evaluatorId);
      } else {
        selectedEvaluators.splice(index, 1);
      }
      
      return {
        ...prev,
        selectedEvaluators
      };
    });
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
        return <Bar data={chartData} options={options} height={400} />;
      case 'line':
        return <Line data={chartData} options={options} height={400} />;
      case 'radar':
        return <Radar data={chartData} options={{ ...options, scales: undefined }} height={400} />;
      case 'pie':
        return <Pie data={chartData} options={{ ...options, scales: undefined }} height={400} />;
      default:
        return <Bar data={chartData} options={options} height={400} />;
    }
  };
  
  // Render raw data table
  const renderRawDataTable = () => {
    if (!resultDetails) return null;
    
    // Filter and sort models
    const filteredModels = availableModels
      .filter(model => {
        // Apply text filter if any
        if (visualizationConfig.filterText) {
          return model.name.toLowerCase().includes(visualizationConfig.filterText.toLowerCase());
        }
        return true;
      })
      .filter(model => visualizationConfig.selectedModels.includes(model.id))
      .sort((a, b) => {
        if (visualizationConfig.sortBy === 'name') {
          return visualizationConfig.sortDirection === 'asc'
            ? a.name.localeCompare(b.name)
            : b.name.localeCompare(a.name);
        } else {
          // Sort by score of first metric
          const metricId = visualizationConfig.metrics[0];
          const scoreA = resultDetails.models[a.id]?.scores?.[metricId]?.average || 0;
          const scoreB = resultDetails.models[b.id]?.scores?.[metricId]?.average || 0;
          
          return visualizationConfig.sortDirection === 'asc'
            ? scoreA - scoreB
            : scoreB - scoreA;
        }
      });
    
    return (
      <div className="mt-4">
        <div className="d-flex justify-content-between align-items-center mb-3">
          <h5>Raw Evaluation Scores</h5>
          <div className="d-flex gap-2">
            <Form.Control
              type="text"
              placeholder="Filter models..."
              value={visualizationConfig.filterText}
              onChange={(e) => handleConfigChange('filterText', e.target.value)}
              style={{ width: '200px' }}
            />
            <Dropdown>
              <Dropdown.Toggle variant="outline-secondary" id="sort-dropdown">
                Sort: {visualizationConfig.sortBy === 'name' ? 'Name' : 'Score'} ({visualizationConfig.sortDirection === 'asc' ? 'Asc' : 'Desc'})
              </Dropdown.Toggle>
              <Dropdown.Menu>
                <Dropdown.Item onClick={() => handleConfigChange('sortBy', 'name')}>Sort by Name</Dropdown.Item>
                <Dropdown.Item onClick={() => handleConfigChange('sortBy', 'score')}>Sort by Score</Dropdown.Item>
                <Dropdown.Divider />
                <Dropdown.Item onClick={() => handleConfigChange('sortDirection', 'asc')}>Ascending</Dropdown.Item>
                <Dropdown.Item onClick={() => handleConfigChange('sortDirection', 'desc')}>Descending</Dropdown.Item>
              </Dropdown.Menu>
            </Dropdown>
          </div>
        </div>
        <Table striped bordered hover size="sm">
          <thead>
            <tr>
              <th>Model</th>
              {availableMetrics
                .filter(metric => visualizationConfig.metrics.includes(metric.id))
                .map(metric => (
                  <th key={metric.id}>{metric.name}</th>
                ))}
            </tr>
          </thead>
          <tbody>
            {filteredModels.map(model => (
              <tr key={model.id}>
                <td>{model.name}</td>
                {availableMetrics
                  .filter(metric => visualizationConfig.metrics.includes(metric.id))
                  .map(metric => {
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
  
  if (loading) {
    return (
      <Container className="mt-4">
        <div className="text-center">
          <Spinner animation="border" role="status">
            <span className="visually-hidden">Loading...</span>
          </Spinner>
          <p className="mt-2">Loading evaluation results...</p>
        </div>
      </Container>
    );
  }
  
  if (error) {
    return (
      <Container className="mt-4">
        <Alert variant="danger">
          <Alert.Heading>Error</Alert.Heading>
          <p>{error}</p>
          <Button variant="outline-danger" onClick={() => navigate('/scripts/evaluation')}>
            Back to Evaluation Hub
          </Button>
        </Alert>
      </Container>
    );
  }
  
  if (!resultDetails) {
    return (
      <Container className="mt-4">
        <Alert variant="info">
          <Alert.Heading>No Results</Alert.Heading>
          <p>No evaluation results found for the specified run ID.</p>
          <Button variant="outline-primary" onClick={() => navigate('/scripts/evaluation')}>
            Back to Evaluation Hub
          </Button>
        </Alert>
      </Container>
    );
  }
  
  return (
    <Container fluid className="mt-4">
      <Row className="mb-4">
        <Col>
          <div className="d-flex justify-content-between align-items-center">
            <h2>Evaluation Results: {resultDetails.run_name}</h2>
            <Button variant="outline-secondary" onClick={() => navigate('/scripts/evaluation')}>
              Back to Evaluation Hub
            </Button>
          </div>
          <p className="text-muted">
            {formatDate(resultDetails.timestamp)} | Dataset: {resultDetails.dataset_type} | Examples: {resultDetails.dataset_size}
          </p>
        </Col>
      </Row>
      
      <Row>
        <Col md={3}>
          <Card className="mb-4">
            <Card.Header>Visualization Settings</Card.Header>
            <Card.Body>
              <Form>
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
                    <option value="evaluator">Compare Evaluators</option>
                  </Form.Select>
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Label>Select Metrics</Form.Label>
                  <div className="border rounded p-2" style={{ maxHeight: '150px', overflowY: 'auto' }}>
                    {availableMetrics.map(metric => (
                      <Form.Check
                        key={metric.id}
                        type="checkbox"
                        id={`metric-${metric.id}`}
                        label={metric.name}
                        checked={visualizationConfig.metrics.includes(metric.id)}
                        onChange={() => handleMetricSelect(metric.id)}
                      />
                    ))}
                  </div>
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Label>Select Models</Form.Label>
                  <div className="border rounded p-2" style={{ maxHeight: '150px', overflowY: 'auto' }}>
                    {availableModels.map(model => (
                      <Form.Check
                        key={model.id}
                        type="checkbox"
                        id={`model-${model.id}`}
                        label={model.name}
                        checked={visualizationConfig.selectedModels.includes(model.id)}
                        onChange={() => handleModelSelect(model.id)}
                      />
                    ))}
                  </div>
                </Form.Group>
                
                {visualizationConfig.compareBy === 'evaluator' && (
                  <Form.Group className="mb-3">
                    <Form.Label>Select Evaluators</Form.Label>
                    <div className="border rounded p-2" style={{ maxHeight: '150px', overflowY: 'auto' }}>
                      {availableEvaluators.map(evaluator => (
                        <Form.Check
                          key={evaluator.id}
                          type="checkbox"
                          id={`evaluator-${evaluator.id}`}
                          label={evaluator.name}
                          checked={visualizationConfig.selectedEvaluators.includes(evaluator.id)}
                          onChange={() => handleEvaluatorSelect(evaluator.id)}
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
              </Form>
            </Card.Body>
          </Card>
        </Col>
        
        <Col md={9}>
          <Card className="mb-4">
            <Card.Header>Visualization</Card.Header>
            <Card.Body>
              <div style={{ height: '400px' }}>
                {renderChart()}
              </div>
              
              {visualizationConfig.showRawData && renderRawDataTable()}
            </Card.Body>
          </Card>
          
          <Card>
            <Card.Header>Example Results</Card.Header>
            <Card.Body>
              <Tabs defaultActiveKey="examples" className="mb-3">
                <Tab eventKey="examples" title="Examples">
                  {availableModels
                    .filter(model => visualizationConfig.selectedModels.includes(model.id))
                    .map(model => {
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
                </Tab>
                
                <Tab eventKey="summary" title="Summary">
                  <Table striped bordered hover>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Average Score</th>
                        <th>Best Metric</th>
                        <th>Worst Metric</th>
                      </tr>
                    </thead>
                    <tbody>
                      {availableModels
                        .filter(model => visualizationConfig.selectedModels.includes(model.id))
                        .map(model => {
                          const modelScores = resultDetails.models[model.id]?.scores || {};
                          const scores = Object.entries(modelScores).map(([metricId, data]) => ({
                            metricId,
                            score: data.average || 0
                          }));
                          
                          const avgScore = scores.reduce((sum, item) => sum + item.score, 0) / (scores.length || 1);
                          const bestMetric = scores.reduce((best, item) => item.score > best.score ? item : best, { score: -1 });
                          const worstMetric = scores.reduce((worst, item) => item.score < worst.score ? item : worst, { score: 2 });
                          
                          return (
                            <tr key={model.id}>
                              <td>{model.name}</td>
                              <td>{avgScore.toFixed(4)}</td>
                              <td>
                                {bestMetric.metricId} ({bestMetric.score.toFixed(4)})
                              </td>
                              <td>
                                {worstMetric.metricId} ({worstMetric.score.toFixed(4)})
                              </td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </Table>
                </Tab>
              </Tabs>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default EvaluationResultsVisualization;
