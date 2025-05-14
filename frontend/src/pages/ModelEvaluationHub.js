import React, { useState, useEffect, useRef } from 'react';
import { Container, Row, Col, Card, Table, Button, Spinner, Alert, Form, Modal, Tabs, Tab, Badge } from 'react-bootstrap';
import axios from 'axios';
import ResultsVisualization from '../components/ResultsVisualization';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:7433';

const ModelEvaluationHub = () => {
  // Results state
  const [evaluationResults, setEvaluationResults] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);
  const [resultDetails, setResultDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // New evaluation run state
  const [showNewEvalModal, setShowNewEvalModal] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [availableEvaluators, setAvailableEvaluators] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [selectedDatasets, setSelectedDatasets] = useState([]);
  const [selectedEvaluators, setSelectedEvaluators] = useState([]);
  const [runName, setRunName] = useState('');
  const [maxExamples, setMaxExamples] = useState(50);

  // Evaluator customization
  const [showEvaluatorCustomization, setShowEvaluatorCustomization] = useState(false);
  const [selectedEvaluatorForCustomization, setSelectedEvaluatorForCustomization] = useState(null);
  const [evaluatorCustomizations, setEvaluatorCustomizations] = useState({});

  // Results visualization
  const [showResultsVisualization, setShowResultsVisualization] = useState(false);
  const [visualizationConfig, setVisualizationConfig] = useState({
    chartType: 'bar',
    compareBy: 'model',
    metrics: ['correctness', 'relevance'],
    showRawData: false
  });

  // RAG-specific configuration
  const [isRagEvaluation, setIsRagEvaluation] = useState(false);
  const [ragTopic, setRagTopic] = useState('metasploit');
  const [ragTestType, setRagTestType] = useState('all');
  const [availableRagTopics, setAvailableRagTopics] = useState([
    'metasploit', 'nmap', 'wireshark', 'burpsuite', 'cybersecurity'
  ]);

  // Active run state
  const [activeRun, setActiveRun] = useState(null);
  const [activeRunStatus, setActiveRunStatus] = useState(null);
  const [activeRunLogs, setActiveRunLogs] = useState([]);
  const [statusPollingInterval, setStatusPollingInterval] = useState(null);
  const [logsPollingInterval, setLogsPollingInterval] = useState(null);
  const [lastLogCount, setLastLogCount] = useState(0);

  // Fetch evaluation results on component mount
  useEffect(() => {
    fetchEvaluationResults();
    fetchAvailableModels();
    fetchAvailableDatasets();
    fetchAvailableEvaluators();
  }, []);

  // Clean up polling intervals on unmount
  useEffect(() => {
    return () => {
      if (statusPollingInterval) clearInterval(statusPollingInterval);
      if (logsPollingInterval) clearInterval(logsPollingInterval);
    };
  }, [statusPollingInterval, logsPollingInterval]);

  // Fetch evaluation results from the API
  const fetchEvaluationResults = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_URL}/api/evaluation/results`);
      setEvaluationResults(response.data);
    } catch (err) {
      console.error('Error fetching evaluation results:', err);
      setError('Failed to fetch evaluation results. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  // Fetch details for a specific evaluation result
  const fetchResultDetails = async (resultId) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_URL}/api/evaluation/result/${resultId}`);
      setResultDetails(response.data);
    } catch (err) {
      console.error(`Error fetching details for result ${resultId}:`, err);
      setError(`Failed to fetch details for result ${resultId}. Please try again later.`);
    } finally {
      setLoading(false);
    }
  };

  // Handle result selection
  const handleResultSelect = (result) => {
    setSelectedResult(result);
    fetchResultDetails(result.id);
  };

  // Format date string
  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch (err) {
      return dateString;
    }
  };

  // Fetch available models
  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/evaluation/models`);
      setAvailableModels(response.data);
    } catch (err) {
      console.error('Error fetching available models:', err);
      // Use default models if API fails
      setAvailableModels([
        {
          id: 'teacher',
          name: 'Llama 3 8B Instruct (Teacher)',
          endpoint: 'http://localhost:8000/v1',
          model_id: 'casperhansen/llama-3-8b-instruct-awq',
          type: 'teacher'
        },
        {
          id: 'student',
          name: 'Phi-3 Mini (Student)',
          endpoint: 'http://localhost:8002/v1',
          model_id: 'microsoft/Phi-3-mini-4k-instruct',
          type: 'student'
        }
      ]);
    }
  };

  // Fetch available datasets
  const fetchAvailableDatasets = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/evaluation/datasets`);
      setAvailableDatasets(response.data);
    } catch (err) {
      console.error('Error fetching available datasets:', err);
      // Use default datasets if API fails
      setAvailableDatasets([
        // Standard datasets
        { id: 'error_suggestion', name: 'Error Suggestion', type: 'error_suggestion', category: 'standard' },
        { id: 'command_extraction', name: 'Command Extraction', type: 'command_extraction', category: 'standard' },
        { id: 'qa', name: 'Question Answering', type: 'qa', category: 'standard' },

        // RAG datasets
        { id: 'rag_standard', name: 'RAG Standard Queries', type: 'rag', category: 'rag', test_type: 'standard' },
        { id: 'rag_noisy', name: 'RAG Noisy Retrieval', type: 'rag', category: 'rag', test_type: 'noisy_retrieval' },
        { id: 'rag_contradictory', name: 'RAG Contradictory Info', type: 'rag', category: 'rag', test_type: 'contradictory_information' },
        { id: 'rag_no_info', name: 'RAG Missing Information', type: 'rag', category: 'rag', test_type: 'information_not_present' },
        { id: 'rag_precision', name: 'RAG Precision Test', type: 'rag', category: 'rag', test_type: 'precision_test' }
      ]);
    }
  };

  // Fetch available evaluators
  const fetchAvailableEvaluators = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/evaluation/evaluators`);
      setAvailableEvaluators(response.data);
    } catch (err) {
      console.error('Error fetching available evaluators:', err);
      // Use default evaluators if API fails
      setAvailableEvaluators([
        // Standard evaluators
        { id: 'string_match', name: 'String Match', type: 'string_distance', category: 'standard' },
        { id: 'embedding_similarity', name: 'Embedding Similarity', type: 'embedding_distance', category: 'standard' },
        { id: 'error_suggestion_judge', name: 'Error Suggestion Judge', type: 'llm_as_judge', category: 'standard' },
        { id: 'command_extraction_judge', name: 'Command Extraction Judge', type: 'llm_as_judge', category: 'standard' },
        { id: 'qa_correctness', name: 'QA Correctness', type: 'llm_as_judge', category: 'standard' },

        // RAG evaluators
        { id: 'retrieval_hit_rate', name: 'Retrieval Hit Rate', type: 'custom', class: 'RetrievalHitRateEvaluator', category: 'rag' },
        { id: 'retrieval_precision', name: 'Retrieval Precision', type: 'custom', class: 'RetrievalPrecisionEvaluator', category: 'rag' },
        { id: 'faithfulness', name: 'Faithfulness', type: 'custom', class: 'FaithfulnessEvaluator', model: 'teacher', category: 'rag' },
        { id: 'contradiction_handling', name: 'Contradiction Handling', type: 'custom', class: 'ContradictionHandlingEvaluator', model: 'teacher', category: 'rag' },
        { id: 'noise_robustness', name: 'Noise Robustness', type: 'custom', class: 'NoiseRobustnessEvaluator', model: 'teacher', category: 'rag' },
        { id: 'no_info_handling', name: 'No Info Handling', type: 'custom', class: 'NoInfoHandlingEvaluator', model: 'teacher', category: 'rag' }
      ]);
    }
  };

  // Render the evaluation results table
  const renderResultsTable = () => {
    if (evaluationResults.length === 0) {
      return <Alert variant="info">No evaluation results found. Run an evaluation to see results here.</Alert>;
    }

    return (
      <Table striped bordered hover>
        <thead>
          <tr>
            <th>Name</th>
            <th>Date</th>
            <th>Dataset Type</th>
            <th>Dataset Size</th>
            <th>Models</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {evaluationResults.map((result) => (
            <tr key={result.id} className={selectedResult?.id === result.id ? 'table-primary' : ''}>
              <td>{result.name}</td>
              <td>{formatDate(result.timestamp)}</td>
              <td>{result.dataset_type}</td>
              <td>{result.dataset_size}</td>
              <td>{result.models?.join(', ')}</td>
              <td>
                <div className="d-flex gap-2">
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={() => handleResultSelect(result)}
                  >
                    View Details
                  </Button>
                  <Button
                    variant="success"
                    size="sm"
                    onClick={() => window.open(`/evaluation/results/${result.id}`, '_blank')}
                  >
                    Advanced Visualization
                  </Button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
    );
  };

  // Handle starting a new evaluation run
  const handleStartEvaluation = async () => {
    try {
      // Validate inputs
      if (selectedModels.length === 0) {
        setError('Please select at least one model');
        return;
      }
      if (selectedDatasets.length === 0) {
        setError('Please select at least one dataset');
        return;
      }
      if (selectedEvaluators.length === 0) {
        setError('Please select at least one evaluator');
        return;
      }

      setLoading(true);
      setError(null);

      // Prepare request data
      const requestData = {
        models: selectedModels,
        datasets: selectedDatasets.map(d => d.id),
        evaluators: selectedEvaluators.map(e => ({
          id: e.id,
          type: e.type,
          class: e.class,
          model: e.model,
          // Include customizations if available
          customization: evaluatorCustomizations[e.id] || {}
        })),
        run_name: runName || `Evaluation Run ${new Date().toLocaleString()}`,
        max_examples: maxExamples
      };

      // Add RAG-specific parameters if this is a RAG evaluation
      if (isRagEvaluation) {
        requestData.is_rag_evaluation = true;
        requestData.rag_topic = ragTopic;
        requestData.rag_test_type = ragTestType;

        // Add test types from selected datasets
        const testTypes = selectedDatasets
          .filter(d => d.category === 'rag')
          .map(d => d.test_type);

        if (testTypes.length > 0) {
          requestData.rag_test_types = testTypes;
        }
      }

      // Send request to start evaluation
      const response = await axios.post(`${API_URL}/api/evaluation/run`, requestData);

      // Close the modal
      setShowNewEvalModal(false);

      // Set active run
      setActiveRun(response.data);

      // Start polling for status and logs
      startStatusPolling(response.data.run_id);
      startLogsPolling(response.data.run_id);

      // Show success message
      setError(null);
    } catch (err) {
      console.error('Error starting evaluation:', err);
      setError(`Failed to start evaluation: ${err.response?.data?.error || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Start polling for run status
  const startStatusPolling = (runId) => {
    // Clear any existing interval
    if (statusPollingInterval) {
      clearInterval(statusPollingInterval);
    }

    // Fetch status immediately
    fetchRunStatus(runId);

    // Set up polling interval (every 5 seconds)
    const interval = setInterval(() => {
      fetchRunStatus(runId);
    }, 5000);

    setStatusPollingInterval(interval);
  };

  // Fetch run status
  const fetchRunStatus = async (runId) => {
    try {
      const response = await axios.get(`${API_URL}/api/evaluation/run/${runId}/status`);
      setActiveRunStatus(response.data);

      // If run is completed or failed, stop polling
      if (['completed', 'failed', 'cancelled'].includes(response.data.status)) {
        if (statusPollingInterval) {
          clearInterval(statusPollingInterval);
          setStatusPollingInterval(null);
        }

        // Refresh results list if completed
        if (response.data.status === 'completed') {
          fetchEvaluationResults();
        }
      }
    } catch (err) {
      console.error(`Error fetching status for run ${runId}:`, err);
    }
  };

  // Start polling for run logs
  const startLogsPolling = (runId) => {
    // Clear any existing interval
    if (logsPollingInterval) {
      clearInterval(logsPollingInterval);
    }

    // Fetch logs immediately
    fetchRunLogs(runId);

    // Set up polling interval (every 3 seconds)
    const interval = setInterval(() => {
      fetchRunLogs(runId);
    }, 3000);

    setLogsPollingInterval(interval);
  };

  // Fetch run logs
  const fetchRunLogs = async (runId) => {
    try {
      const response = await axios.get(`${API_URL}/api/evaluation/run/${runId}/logs`, {
        params: {
          start: lastLogCount,
          limit: 100
        }
      });

      // Append new logs
      if (response.data.logs.length > 0) {
        setActiveRunLogs(prevLogs => [...prevLogs, ...response.data.logs]);
        setLastLogCount(prevCount => prevCount + response.data.logs.length);
      }

      // If run is completed or failed, stop polling
      if (['completed', 'failed', 'cancelled'].includes(response.data.status)) {
        if (logsPollingInterval) {
          clearInterval(logsPollingInterval);
          setLogsPollingInterval(null);
        }
      }
    } catch (err) {
      console.error(`Error fetching logs for run ${runId}:`, err);
    }
  };

  // Cancel active run
  const cancelActiveRun = async () => {
    if (!activeRun) return;

    try {
      await axios.post(`${API_URL}/api/evaluation/run/${activeRun.run_id}/cancel`);
      fetchRunStatus(activeRun.run_id);
    } catch (err) {
      console.error(`Error cancelling run ${activeRun.run_id}:`, err);
      setError(`Failed to cancel run: ${err.response?.data?.error || err.message}`);
    }
  };

  // Reset active run
  const resetActiveRun = () => {
    setActiveRun(null);
    setActiveRunStatus(null);
    setActiveRunLogs([]);
    setLastLogCount(0);

    if (statusPollingInterval) {
      clearInterval(statusPollingInterval);
      setStatusPollingInterval(null);
    }

    if (logsPollingInterval) {
      clearInterval(logsPollingInterval);
      setLogsPollingInterval(null);
    }
  };

  // Handle model selection
  const handleModelSelect = (model) => {
    setSelectedModels(prevSelected => {
      const isSelected = prevSelected.some(m => m.id === model.id);
      if (isSelected) {
        return prevSelected.filter(m => m.id !== model.id);
      } else {
        return [...prevSelected, model];
      }
    });
  };

  // Handle dataset selection
  const handleDatasetSelect = (dataset) => {
    setSelectedDatasets(prevSelected => {
      const isSelected = prevSelected.some(d => d.id === dataset.id);
      let newSelected;

      if (isSelected) {
        newSelected = prevSelected.filter(d => d.id !== dataset.id);
      } else {
        newSelected = [...prevSelected, dataset];
      }

      // Check if any RAG datasets are selected
      const hasRagDatasets = newSelected.some(d => d.category === 'rag');
      setIsRagEvaluation(hasRagDatasets);

      // If selecting a RAG dataset, automatically select appropriate RAG evaluators
      if (!isSelected && dataset.category === 'rag') {
        // Find RAG evaluators that aren't already selected
        const ragEvaluators = availableEvaluators.filter(e =>
          e.category === 'rag' && !selectedEvaluators.some(se => se.id === e.id)
        );

        // Add them to selected evaluators
        if (ragEvaluators.length > 0) {
          setSelectedEvaluators(prev => [...prev, ...ragEvaluators]);
        }
      }

      return newSelected;
    });
  };

  // Handle evaluator selection
  const handleEvaluatorSelect = (evaluator) => {
    setSelectedEvaluators(prevSelected => {
      const isSelected = prevSelected.some(e => e.id === evaluator.id);
      if (isSelected) {
        // Remove evaluator and its customizations
        const newEvaluatorCustomizations = { ...evaluatorCustomizations };
        delete newEvaluatorCustomizations[evaluator.id];
        setEvaluatorCustomizations(newEvaluatorCustomizations);

        return prevSelected.filter(e => e.id !== evaluator.id);
      } else {
        // Add evaluator with default customizations
        const defaultCustomization = getDefaultCustomizationForEvaluator(evaluator);
        if (defaultCustomization) {
          setEvaluatorCustomizations(prev => ({
            ...prev,
            [evaluator.id]: defaultCustomization
          }));
        }

        return [...prevSelected, evaluator];
      }
    });
  };

  // Get default customization options for an evaluator
  const getDefaultCustomizationForEvaluator = (evaluator) => {
    if (evaluator.type === 'llm_as_judge') {
      return {
        criteria: ['correctness'],
        temperature: 0.2,
        detailed_feedback: true
      };
    } else if (evaluator.type === 'embedding_distance') {
      return {
        embedding_model: 'all-MiniLM-L6-v2',
        distance_metric: 'cosine'
      };
    } else if (evaluator.type === 'custom' && evaluator.category === 'rag') {
      if (evaluator.id === 'retrieval_hit_rate' || evaluator.id === 'retrieval_precision') {
        return {
          min_score_threshold: 0.5,
          consider_partial_matches: true
        };
      } else if (evaluator.id === 'faithfulness') {
        return {
          strictness_level: 'medium',
          check_hallucinations: true
        };
      } else if (evaluator.id === 'contradiction_handling') {
        return {
          contradiction_detection_threshold: 0.7,
          resolution_quality_threshold: 0.6
        };
      } else if (evaluator.id === 'noise_robustness') {
        return {
          noise_tolerance_threshold: 0.6,
          noise_ratio: 0.3
        };
      } else if (evaluator.id === 'no_info_handling') {
        return {
          honesty_threshold: 0.8,
          penalize_fabrication: true
        };
      }
    }

    return null;
  };

  // Handle customization for a specific evaluator
  const handleCustomizeEvaluator = (evaluator) => {
    setSelectedEvaluatorForCustomization(evaluator);
    setShowEvaluatorCustomization(true);
  };

  // Update customization for an evaluator
  const updateEvaluatorCustomization = (evaluatorId, customization) => {
    setEvaluatorCustomizations(prev => ({
      ...prev,
      [evaluatorId]: customization
    }));
  };

  // Render the evaluator customization modal
  const renderEvaluatorCustomizationModal = () => {
    if (!selectedEvaluatorForCustomization) return null;

    const evaluator = selectedEvaluatorForCustomization;
    const customization = evaluatorCustomizations[evaluator.id] || {};

    const handleCustomizationChange = (key, value) => {
      const updatedCustomization = {
        ...customization,
        [key]: value
      };
      updateEvaluatorCustomization(evaluator.id, updatedCustomization);
    };

    return (
      <Modal
        show={showEvaluatorCustomization}
        onHide={() => setShowEvaluatorCustomization(false)}
        size="lg"
        centered
      >
        <Modal.Header closeButton>
          <Modal.Title>Customize Evaluator: {evaluator.name}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p className="text-muted mb-4">
            Customize the parameters for this evaluator to fine-tune its behavior.
          </p>

          {evaluator.type === 'llm_as_judge' && (
            <div>
              <Form.Group className="mb-3">
                <Form.Label>Evaluation Criteria</Form.Label>
                <div className="mb-2">
                  <Form.Check
                    type="checkbox"
                    id="criteria-correctness"
                    label="Correctness"
                    checked={customization.criteria?.includes('correctness')}
                    onChange={(e) => {
                      const criteria = [...(customization.criteria || [])];
                      if (e.target.checked) {
                        if (!criteria.includes('correctness')) criteria.push('correctness');
                      } else {
                        const index = criteria.indexOf('correctness');
                        if (index !== -1) criteria.splice(index, 1);
                      }
                      handleCustomizationChange('criteria', criteria);
                    }}
                  />
                  <Form.Check
                    type="checkbox"
                    id="criteria-relevance"
                    label="Relevance"
                    checked={customization.criteria?.includes('relevance')}
                    onChange={(e) => {
                      const criteria = [...(customization.criteria || [])];
                      if (e.target.checked) {
                        if (!criteria.includes('relevance')) criteria.push('relevance');
                      } else {
                        const index = criteria.indexOf('relevance');
                        if (index !== -1) criteria.splice(index, 1);
                      }
                      handleCustomizationChange('criteria', criteria);
                    }}
                  />
                  <Form.Check
                    type="checkbox"
                    id="criteria-completeness"
                    label="Completeness"
                    checked={customization.criteria?.includes('completeness')}
                    onChange={(e) => {
                      const criteria = [...(customization.criteria || [])];
                      if (e.target.checked) {
                        if (!criteria.includes('completeness')) criteria.push('completeness');
                      } else {
                        const index = criteria.indexOf('completeness');
                        if (index !== -1) criteria.splice(index, 1);
                      }
                      handleCustomizationChange('criteria', criteria);
                    }}
                  />
                  <Form.Check
                    type="checkbox"
                    id="criteria-conciseness"
                    label="Conciseness"
                    checked={customization.criteria?.includes('conciseness')}
                    onChange={(e) => {
                      const criteria = [...(customization.criteria || [])];
                      if (e.target.checked) {
                        if (!criteria.includes('conciseness')) criteria.push('conciseness');
                      } else {
                        const index = criteria.indexOf('conciseness');
                        if (index !== -1) criteria.splice(index, 1);
                      }
                      handleCustomizationChange('criteria', criteria);
                    }}
                  />
                </div>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Temperature</Form.Label>
                <Form.Control
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={customization.temperature || 0.2}
                  onChange={(e) => handleCustomizationChange('temperature', parseFloat(e.target.value))}
                />
                <Form.Text className="text-muted">
                  Controls randomness in the judge's evaluation (0.0-1.0).
                </Form.Text>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Check
                  type="checkbox"
                  id="detailed-feedback"
                  label="Provide Detailed Feedback"
                  checked={customization.detailed_feedback}
                  onChange={(e) => handleCustomizationChange('detailed_feedback', e.target.checked)}
                />
                <Form.Text className="text-muted">
                  If enabled, the judge will provide detailed reasoning for its scores.
                </Form.Text>
              </Form.Group>
            </div>
          )}

          {evaluator.type === 'embedding_distance' && (
            <div>
              <Form.Group className="mb-3">
                <Form.Label>Embedding Model</Form.Label>
                <Form.Select
                  value={customization.embedding_model || 'all-MiniLM-L6-v2'}
                  onChange={(e) => handleCustomizationChange('embedding_model', e.target.value)}
                >
                  <option value="all-MiniLM-L6-v2">all-MiniLM-L6-v2</option>
                  <option value="all-mpnet-base-v2">all-mpnet-base-v2</option>
                  <option value="all-distilroberta-v1">all-distilroberta-v1</option>
                </Form.Select>
                <Form.Text className="text-muted">
                  The model used to generate embeddings for similarity comparison.
                </Form.Text>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Distance Metric</Form.Label>
                <Form.Select
                  value={customization.distance_metric || 'cosine'}
                  onChange={(e) => handleCustomizationChange('distance_metric', e.target.value)}
                >
                  <option value="cosine">Cosine Similarity</option>
                  <option value="euclidean">Euclidean Distance</option>
                  <option value="dot_product">Dot Product</option>
                </Form.Select>
                <Form.Text className="text-muted">
                  The metric used to calculate distance between embeddings.
                </Form.Text>
              </Form.Group>
            </div>
          )}

          {evaluator.type === 'custom' && evaluator.category === 'rag' && (
            <div>
              {(evaluator.id === 'retrieval_hit_rate' || evaluator.id === 'retrieval_precision') && (
                <div>
                  <Form.Group className="mb-3">
                    <Form.Label>Minimum Score Threshold</Form.Label>
                    <Form.Control
                      type="number"
                      min="0"
                      max="1"
                      step="0.1"
                      value={customization.min_score_threshold || 0.5}
                      onChange={(e) => handleCustomizationChange('min_score_threshold', parseFloat(e.target.value))}
                    />
                    <Form.Text className="text-muted">
                      Minimum score required for a retrieval to be considered successful.
                    </Form.Text>
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Check
                      type="checkbox"
                      id="consider-partial-matches"
                      label="Consider Partial Matches"
                      checked={customization.consider_partial_matches}
                      onChange={(e) => handleCustomizationChange('consider_partial_matches', e.target.checked)}
                    />
                    <Form.Text className="text-muted">
                      If enabled, partial matches will be considered in the evaluation.
                    </Form.Text>
                  </Form.Group>
                </div>
              )}

              {evaluator.id === 'faithfulness' && (
                <div>
                  <Form.Group className="mb-3">
                    <Form.Label>Strictness Level</Form.Label>
                    <Form.Select
                      value={customization.strictness_level || 'medium'}
                      onChange={(e) => handleCustomizationChange('strictness_level', e.target.value)}
                    >
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                    </Form.Select>
                    <Form.Text className="text-muted">
                      How strictly to evaluate faithfulness to the retrieved context.
                    </Form.Text>
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Check
                      type="checkbox"
                      id="check-hallucinations"
                      label="Check for Hallucinations"
                      checked={customization.check_hallucinations}
                      onChange={(e) => handleCustomizationChange('check_hallucinations', e.target.checked)}
                    />
                    <Form.Text className="text-muted">
                      If enabled, the evaluator will specifically check for hallucinated content.
                    </Form.Text>
                  </Form.Group>
                </div>
              )}

              {evaluator.id === 'contradiction_handling' && (
                <div>
                  <Form.Group className="mb-3">
                    <Form.Label>Contradiction Detection Threshold</Form.Label>
                    <Form.Control
                      type="number"
                      min="0"
                      max="1"
                      step="0.1"
                      value={customization.contradiction_detection_threshold || 0.7}
                      onChange={(e) => handleCustomizationChange('contradiction_detection_threshold', parseFloat(e.target.value))}
                    />
                    <Form.Text className="text-muted">
                      Threshold for detecting contradictions in the retrieved context.
                    </Form.Text>
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>Resolution Quality Threshold</Form.Label>
                    <Form.Control
                      type="number"
                      min="0"
                      max="1"
                      step="0.1"
                      value={customization.resolution_quality_threshold || 0.6}
                      onChange={(e) => handleCustomizationChange('resolution_quality_threshold', parseFloat(e.target.value))}
                    />
                    <Form.Text className="text-muted">
                      Threshold for the quality of contradiction resolution.
                    </Form.Text>
                  </Form.Group>
                </div>
              )}

              {evaluator.id === 'noise_robustness' && (
                <div>
                  <Form.Group className="mb-3">
                    <Form.Label>Noise Tolerance Threshold</Form.Label>
                    <Form.Control
                      type="number"
                      min="0"
                      max="1"
                      step="0.1"
                      value={customization.noise_tolerance_threshold || 0.6}
                      onChange={(e) => handleCustomizationChange('noise_tolerance_threshold', parseFloat(e.target.value))}
                    />
                    <Form.Text className="text-muted">
                      Threshold for tolerance to noise in the retrieved context.
                    </Form.Text>
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>Noise Ratio</Form.Label>
                    <Form.Control
                      type="number"
                      min="0"
                      max="1"
                      step="0.1"
                      value={customization.noise_ratio || 0.3}
                      onChange={(e) => handleCustomizationChange('noise_ratio', parseFloat(e.target.value))}
                    />
                    <Form.Text className="text-muted">
                      Ratio of noise to relevant information in the test.
                    </Form.Text>
                  </Form.Group>
                </div>
              )}

              {evaluator.id === 'no_info_handling' && (
                <div>
                  <Form.Group className="mb-3">
                    <Form.Label>Honesty Threshold</Form.Label>
                    <Form.Control
                      type="number"
                      min="0"
                      max="1"
                      step="0.1"
                      value={customization.honesty_threshold || 0.8}
                      onChange={(e) => handleCustomizationChange('honesty_threshold', parseFloat(e.target.value))}
                    />
                    <Form.Text className="text-muted">
                      Threshold for honesty in acknowledging missing information.
                    </Form.Text>
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Check
                      type="checkbox"
                      id="penalize-fabrication"
                      label="Penalize Fabrication"
                      checked={customization.penalize_fabrication}
                      onChange={(e) => handleCustomizationChange('penalize_fabrication', e.target.checked)}
                    />
                    <Form.Text className="text-muted">
                      If enabled, the evaluator will heavily penalize fabricated answers.
                    </Form.Text>
                  </Form.Group>
                </div>
              )}
            </div>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowEvaluatorCustomization(false)}>
            Close
          </Button>
          <Button
            variant="primary"
            onClick={() => setShowEvaluatorCustomization(false)}
          >
            Save Customization
          </Button>
        </Modal.Footer>
      </Modal>
    );
  };

  // Render the new evaluation modal
  const renderNewEvaluationModal = () => {
    return (
      <Modal
        show={showNewEvalModal}
        onHide={() => setShowNewEvalModal(false)}
        size="lg"
        centered
      >
        <Modal.Header closeButton>
          <Modal.Title>Start New Evaluation</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form>
            <Form.Group className="mb-3">
              <Form.Label>Run Name</Form.Label>
              <Form.Control
                type="text"
                placeholder="Enter a name for this evaluation run"
                value={runName}
                onChange={(e) => setRunName(e.target.value)}
              />
              <Form.Text className="text-muted">
                A descriptive name helps identify this run in the results list.
              </Form.Text>
            </Form.Group>

            <Form.Group className="mb-3">
              <Form.Label>Max Examples</Form.Label>
              <Form.Control
                type="number"
                min="1"
                max="1000"
                value={maxExamples}
                onChange={(e) => setMaxExamples(parseInt(e.target.value) || 50)}
              />
              <Form.Text className="text-muted">
                Maximum number of examples to evaluate from each dataset.
              </Form.Text>
            </Form.Group>

            <Tabs defaultActiveKey="models" className="mb-3">
              <Tab eventKey="models" title="Models">
                <div className="mb-3">
                  <Form.Label>Select Models to Evaluate</Form.Label>
                  <div className="d-flex flex-wrap gap-2 mb-2">
                    {selectedModels.map(model => (
                      <Badge
                        key={model.id}
                        bg="primary"
                        className="p-2 d-flex align-items-center"
                      >
                        {model.name}
                        <Button
                          variant="link"
                          className="p-0 ms-2 text-white"
                          onClick={() => handleModelSelect(model)}
                        >
                          ×
                        </Button>
                      </Badge>
                    ))}
                  </div>
                  <div className="border rounded p-2" style={{ maxHeight: '200px', overflowY: 'auto' }}>
                    {availableModels.map(model => (
                      <Form.Check
                        key={model.id}
                        type="checkbox"
                        id={`model-${model.id}`}
                        label={`${model.name} (${model.type})`}
                        checked={selectedModels.some(m => m.id === model.id)}
                        onChange={() => handleModelSelect(model)}
                        className="mb-2"
                      />
                    ))}
                  </div>
                </div>
              </Tab>

              <Tab eventKey="datasets" title="Datasets">
                <div className="mb-3">
                  <Form.Label>Select Datasets to Evaluate</Form.Label>
                  <div className="d-flex flex-wrap gap-2 mb-2">
                    {selectedDatasets.map(dataset => (
                      <Badge
                        key={dataset.id}
                        bg={dataset.category === 'rag' ? 'warning' : 'success'}
                        className="p-2 d-flex align-items-center"
                      >
                        {dataset.name}
                        <Button
                          variant="link"
                          className="p-0 ms-2 text-white"
                          onClick={() => handleDatasetSelect(dataset)}
                        >
                          ×
                        </Button>
                      </Badge>
                    ))}
                  </div>

                  {/* Group datasets by category */}
                  <div className="border rounded p-2" style={{ maxHeight: '200px', overflowY: 'auto' }}>
                    <div className="mb-2">
                      <strong>Standard Datasets</strong>
                      {availableDatasets
                        .filter(dataset => dataset.category === 'standard' || !dataset.category)
                        .map(dataset => (
                          <Form.Check
                            key={dataset.id}
                            type="checkbox"
                            id={`dataset-${dataset.id}`}
                            label={`${dataset.name} (${dataset.example_count || '?'} examples)`}
                            checked={selectedDatasets.some(d => d.id === dataset.id)}
                            onChange={() => handleDatasetSelect(dataset)}
                            className="mb-2 ms-3"
                          />
                        ))
                      }
                    </div>

                    <div>
                      <strong>RAG Datasets</strong>
                      {availableDatasets
                        .filter(dataset => dataset.category === 'rag')
                        .map(dataset => (
                          <Form.Check
                            key={dataset.id}
                            type="checkbox"
                            id={`dataset-${dataset.id}`}
                            label={`${dataset.name} (${dataset.test_type})`}
                            checked={selectedDatasets.some(d => d.id === dataset.id)}
                            onChange={() => handleDatasetSelect(dataset)}
                            className="mb-2 ms-3"
                          />
                        ))
                      }
                    </div>
                  </div>
                </div>
              </Tab>

              <Tab eventKey="evaluators" title="Evaluators">
                <div className="mb-3">
                  <Form.Label>Select Evaluators to Use</Form.Label>
                  <div className="d-flex flex-wrap gap-2 mb-2">
                    {selectedEvaluators.map(evaluator => (
                      <Badge
                        key={evaluator.id}
                        bg={evaluator.category === 'rag' ? 'warning' : 'info'}
                        className="p-2 d-flex align-items-center"
                      >
                        {evaluator.name}
                        {/* Customize button */}
                        {(evaluator.type === 'llm_as_judge' ||
                          evaluator.type === 'embedding_distance' ||
                          (evaluator.type === 'custom' && evaluator.category === 'rag')) && (
                          <Button
                            variant="link"
                            className="p-0 ms-1 text-white"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleCustomizeEvaluator(evaluator);
                            }}
                            title="Customize evaluator"
                          >
                            <i className="bi bi-gear-fill" style={{ fontSize: '0.8rem' }}></i>
                          </Button>
                        )}
                        {/* Remove button */}
                        <Button
                          variant="link"
                          className="p-0 ms-1 text-white"
                          onClick={() => handleEvaluatorSelect(evaluator)}
                          title="Remove evaluator"
                        >
                          ×
                        </Button>
                      </Badge>
                    ))}
                  </div>

                  {/* Group evaluators by category */}
                  <div className="border rounded p-2" style={{ maxHeight: '200px', overflowY: 'auto' }}>
                    <div className="mb-2">
                      <strong>Standard Evaluators</strong>
                      {availableEvaluators
                        .filter(evaluator => evaluator.category === 'standard' || !evaluator.category)
                        .map(evaluator => (
                          <Form.Check
                            key={evaluator.id}
                            type="checkbox"
                            id={`evaluator-${evaluator.id}`}
                            label={`${evaluator.name} (${evaluator.type})`}
                            checked={selectedEvaluators.some(e => e.id === evaluator.id)}
                            onChange={() => handleEvaluatorSelect(evaluator)}
                            className="mb-2 ms-3"
                          />
                        ))
                      }
                    </div>

                    <div>
                      <strong>RAG Evaluators</strong>
                      {availableEvaluators
                        .filter(evaluator => evaluator.category === 'rag')
                        .map(evaluator => (
                          <Form.Check
                            key={evaluator.id}
                            type="checkbox"
                            id={`evaluator-${evaluator.id}`}
                            label={`${evaluator.name} (${evaluator.type})`}
                            checked={selectedEvaluators.some(e => e.id === evaluator.id)}
                            onChange={() => handleEvaluatorSelect(evaluator)}
                            className="mb-2 ms-3"
                          />
                        ))
                      }
                    </div>
                  </div>
                </div>
              </Tab>

              {/* RAG Configuration Tab - Only shown when RAG datasets are selected */}
              {isRagEvaluation && (
                <Tab eventKey="rag_config" title="RAG Config">
                  <div className="mb-3">
                    <Form.Label>RAG Topic</Form.Label>
                    <Form.Select
                      value={ragTopic}
                      onChange={(e) => setRagTopic(e.target.value)}
                    >
                      {availableRagTopics.map(topic => (
                        <option key={topic} value={topic}>{topic}</option>
                      ))}
                    </Form.Select>
                    <Form.Text className="text-muted">
                      Select the topic for RAG evaluation. This determines which knowledge base will be used.
                    </Form.Text>
                  </div>

                  <div className="mb-3">
                    <Form.Label>Test Type</Form.Label>
                    <Form.Select
                      value={ragTestType}
                      onChange={(e) => setRagTestType(e.target.value)}
                    >
                      <option value="all">All Test Types</option>
                      <option value="standard">Standard Queries</option>
                      <option value="noisy_retrieval">Noisy Retrieval</option>
                      <option value="contradictory_information">Contradictory Information</option>
                      <option value="information_not_present">Missing Information</option>
                      <option value="precision_test">Precision Test</option>
                    </Form.Select>
                    <Form.Text className="text-muted">
                      Select the test type for RAG evaluation. This determines what kind of challenges will be presented to the model.
                    </Form.Text>
                  </div>

                  <Alert variant="info">
                    <Alert.Heading>RAG Evaluation Information</Alert.Heading>
                    <p>
                      RAG evaluation tests how well models can use retrieved information to answer questions.
                      Different test types evaluate different aspects of RAG performance:
                    </p>
                    <ul>
                      <li><strong>Standard Queries:</strong> Basic retrieval and answer generation</li>
                      <li><strong>Noisy Retrieval:</strong> Handling irrelevant information in context</li>
                      <li><strong>Contradictory Information:</strong> Resolving conflicting information</li>
                      <li><strong>Missing Information:</strong> Acknowledging when information is not available</li>
                      <li><strong>Precision Test:</strong> Retrieving exactly the right information</li>
                    </ul>
                  </Alert>
                </Tab>
              )}
            </Tabs>
          </Form>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowNewEvalModal(false)}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={handleStartEvaluation}
            disabled={loading || selectedModels.length === 0 || selectedDatasets.length === 0 || selectedEvaluators.length === 0}
          >
            {loading ? (
              <>
                <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" />
                <span className="ms-2">Starting...</span>
              </>
            ) : (
              'Start Evaluation'
            )}
          </Button>
        </Modal.Footer>
      </Modal>
    );
  };

  // Render active run status
  const renderActiveRunStatus = () => {
    if (!activeRun || !activeRunStatus) return null;

    const statusVariant = {
      'queued': 'info',
      'running': 'primary',
      'completed': 'success',
      'failed': 'danger',
      'cancelled': 'warning'
    }[activeRunStatus.status] || 'secondary';

    return (
      <Card className="mb-4">
        <Card.Header className="d-flex justify-content-between align-items-center">
          <h5 className="mb-0">
            Active Evaluation Run
            <Badge bg={statusVariant} className="ms-2">
              {activeRunStatus.status.toUpperCase()}
            </Badge>
          </h5>
          <div>
            {activeRunStatus.status === 'running' && (
              <Button
                variant="outline-danger"
                size="sm"
                onClick={cancelActiveRun}
                className="me-2"
              >
                Cancel Run
              </Button>
            )}
            <Button
              variant="outline-secondary"
              size="sm"
              onClick={resetActiveRun}
            >
              Dismiss
            </Button>
          </div>
        </Card.Header>
        <Card.Body>
          <Row>
            <Col md={6}>
              <p><strong>Run ID:</strong> {activeRunStatus.run_id}</p>
              <p><strong>Started:</strong> {formatDate(activeRunStatus.start_time)}</p>
              {activeRunStatus.end_time && (
                <p><strong>Ended:</strong> {formatDate(activeRunStatus.end_time)}</p>
              )}
              <p>
                <strong>Datasets:</strong> {activeRunStatus.datasets?.join(', ') || 'None'}
              </p>
              {activeRunStatus.results?.length > 0 && (
                <div>
                  <strong>Results:</strong>
                  <ul className="mb-0">
                    {activeRunStatus.results.map((result, index) => (
                      <li key={index}>
                        {result.dataset}: {result.result_file}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </Col>
            <Col md={6}>
              <div className="border rounded p-2 bg-light" style={{ height: '200px', overflowY: 'auto' }}>
                {activeRunLogs.length === 0 ? (
                  <p className="text-muted">No logs available yet...</p>
                ) : (
                  activeRunLogs.map((log, index) => (
                    <div key={index} className="log-entry small">
                      {log}
                    </div>
                  ))
                )}
              </div>
            </Col>
          </Row>
        </Card.Body>
      </Card>
    );
  };

  // Handle visualization config change
  const handleVisualizationConfigChange = (newConfig) => {
    setVisualizationConfig(newConfig);
  };

  // Toggle results visualization
  const toggleResultsVisualization = () => {
    setShowResultsVisualization(!showResultsVisualization);
  };

  // Render the results visualization
  const renderResultsVisualization = () => {
    if (!selectedResult || !resultDetails) {
      return <Alert variant="info">Select an evaluation result to visualize.</Alert>;
    }

    return (
      <ResultsVisualization
        resultDetails={resultDetails}
        visualizationConfig={visualizationConfig}
        onConfigChange={handleVisualizationConfigChange}
      />
    );
  };

  // Render the result details
  const renderResultDetails = () => {
    if (!selectedResult || !resultDetails) {
      return <Alert variant="info">Select an evaluation result to view details.</Alert>;
    }

    return (
      <Card>
        <Card.Header className="d-flex justify-content-between align-items-center">
          <div>
            <h4>{resultDetails.run_name}</h4>
            <div className="text-muted">
              {formatDate(resultDetails.timestamp)} | Dataset: {resultDetails.dataset_type} | Examples: {resultDetails.dataset_size}
            </div>
          </div>
          <Button
            variant="outline-primary"
            onClick={toggleResultsVisualization}
          >
            {showResultsVisualization ? 'Hide Visualization' : 'Show Visualization'}
          </Button>
        </Card.Header>
        <Card.Body>
          {/* Visualization Section */}
          {showResultsVisualization && (
            <div className="mb-4">
              {renderResultsVisualization()}
            </div>
          )}

          {/* Summary Section */}
          {!showResultsVisualization && resultDetails.summary && (
            <div className="mb-4">
              <h5>Summary</h5>
              <Table striped bordered hover size="sm">
                <thead>
                  <tr>
                    <th>Metric</th>
                    {Object.keys(resultDetails.models || {}).map(modelId => (
                      <th key={modelId}>{resultDetails.models[modelId].name || modelId}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(resultDetails.evaluator_scores || {}).map(([evaluatorId, scores]) => (
                    <tr key={evaluatorId}>
                      <td>{evaluatorId}</td>
                      {Object.keys(resultDetails.models || {}).map(modelId => (
                        <td key={modelId}>
                          {typeof scores[modelId] === 'number'
                            ? scores[modelId].toFixed(4)
                            : scores[modelId] || 'N/A'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </Table>
            </div>
          )}

          {/* Model Comparison Section */}
          {!showResultsVisualization && (
            <div className="mb-4">
              <h5>Model Comparison</h5>
              <div className="d-flex flex-wrap">
                {Object.entries(resultDetails.models || {}).map(([modelId, modelData]) => (
                  <Card key={modelId} className="m-2" style={{ width: '18rem' }}>
                    <Card.Header>{modelData.name || modelId}</Card.Header>
                    <Card.Body>
                      <Table size="sm">
                        <thead>
                          <tr>
                            <th>Metric</th>
                            <th>Score</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(modelData.scores || {}).map(([evalId, evalData]) => (
                            <tr key={evalId}>
                              <td>{evalId}</td>
                              <td>{evalData.average?.toFixed(4) || 'N/A'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </Card.Body>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* Example Results Section */}
          {!showResultsVisualization && (
            <div>
              <h5>Example Results</h5>
              <div className="example-results">
                {Object.entries(resultDetails.models || {}).map(([modelId, modelData]) => (
                  modelData.examples && modelData.examples.length > 0 && (
                    <div key={modelId} className="mb-4">
                      <h6>{modelData.name || modelId} - First 3 Examples</h6>
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
                          {modelData.examples.slice(0, 3).map((example, index) => (
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
                  )
                ))}
              </div>
            </div>
          )}
        </Card.Body>
      </Card>
    );
  };

  return (
    <Container fluid className="py-4">
      <Row className="mb-4">
        <Col>
          <h2>Model Evaluation Hub</h2>
          <p className="text-muted">
            View and compare evaluation results for different models and datasets.
          </p>
        </Col>
        <Col xs="auto">
          <Button
            variant="primary"
            onClick={() => setShowNewEvalModal(true)}
            className="me-2"
          >
            Start New Evaluation
          </Button>
          <Button
            variant="outline-primary"
            onClick={fetchEvaluationResults}
            disabled={loading}
          >
            {loading ? (
              <>
                <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" />
                <span className="ms-2">Loading...</span>
              </>
            ) : (
              'Refresh Results'
            )}
          </Button>
        </Col>
      </Row>

      {error && (
        <Row className="mb-4">
          <Col>
            <Alert variant="danger">{error}</Alert>
          </Col>
        </Row>
      )}

      {/* Active Run Status */}
      {activeRun && renderActiveRunStatus()}

      <Row>
        <Col md={12} lg={12} className="mb-4">
          <Card>
            <Card.Header>Evaluation Results</Card.Header>
            <Card.Body>
              {loading && !selectedResult ? (
                <div className="text-center py-4">
                  <Spinner animation="border" role="status">
                    <span className="visually-hidden">Loading...</span>
                  </Spinner>
                </div>
              ) : (
                renderResultsTable()
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {selectedResult && (
        <Row>
          <Col md={12} lg={12}>
            <Card>
              <Card.Header>Result Details</Card.Header>
              <Card.Body>
                {loading ? (
                  <div className="text-center py-4">
                    <Spinner animation="border" role="status">
                      <span className="visually-hidden">Loading...</span>
                    </Spinner>
                  </div>
                ) : (
                  renderResultDetails()
                )}
              </Card.Body>
            </Card>
          </Col>
        </Row>
      )}

      {/* New Evaluation Modal */}
      {renderNewEvaluationModal()}

      {/* Evaluator Customization Modal */}
      {renderEvaluatorCustomizationModal()}
    </Container>
  );
};

export default ModelEvaluationHub;
