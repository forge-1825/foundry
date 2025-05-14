import React, { useState, useEffect, useRef } from 'react';
import { Container, Row, Col, Card, Button, Table } from 'react-bootstrap';
import { Line, Bar } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';
import axios from 'axios';
import { API_BASE_URL } from '../services/api';

// Register Chart.js components
Chart.register(...registerables);

const NoveltyInsights = () => {
  // State for novelty data
  const [noveltyData, setNoveltyData] = useState({
    timeline: [],
    mostNovel: [],
    leastNovel: [],
    statistics: {
      total_states: 0,
      total_visits: 0,
      avg_visits_per_state: 0,
      max_visits: 0,
      min_visits: 0,
      unique_states: 0
    },
    recentLogs: []
  });

  // State for loading status
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Refs for charts
  const timelineChartRef = useRef(null);
  const distributionChartRef = useRef(null);

  // Function to format novelty score
  const formatNoveltyScore = (score) => {
    return parseFloat(score).toFixed(4);
  };

  // Function to get novelty class based on score
  const getNoveltyClass = (score) => {
    if (score >= 0.7) return 'text-success';
    if (score >= 0.3) return 'text-warning';
    return 'text-danger';
  };

  // Function to fetch novelty data
  const fetchNoveltyData = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/api/novelty/data`);
      setNoveltyData(response.data);
      setError(null);
    } catch (err) {
      console.error('Error fetching novelty data:', err);
      setError('Failed to fetch novelty data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  // Function to download logs
  const downloadLogs = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/novelty/logs?max_lines=1000`);
      const blob = new Blob([response.data.join('\n')], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'curiosity_logs.txt';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Error downloading logs:', err);
      setError('Failed to download logs. Please try again later.');
    }
  };

  // Prepare timeline chart data
  const timelineChartData = {
    labels: noveltyData.timeline.map(item => {
      const date = new Date(item.timestamp);
      return date.toLocaleTimeString();
    }),
    datasets: [{
      label: 'Novelty Score',
      data: noveltyData.timeline.map(item => item.novelty_score),
      borderColor: 'rgba(75, 192, 192, 1)',
      backgroundColor: noveltyData.timeline.map(item => 
        item.success ? 'rgba(75, 192, 192, 0.2)' : 'rgba(255, 99, 132, 0.2)'
      ),
      borderWidth: 1,
      pointBackgroundColor: noveltyData.timeline.map(item => 
        item.success ? 'rgba(75, 192, 192, 1)' : 'rgba(255, 99, 132, 1)'
      ),
      pointRadius: 5,
      pointHoverRadius: 7
    }]
  };

  // Prepare distribution chart data
  const allStates = [...noveltyData.mostNovel, ...noveltyData.leastNovel]
    .sort((a, b) => b.novelty_score - a.novelty_score);

  const distributionChartData = {
    labels: allStates.map(item => item.state.substring(0, 8) + '...'),
    datasets: [{
      label: 'Novelty Score',
      data: allStates.map(item => item.novelty_score),
      backgroundColor: 'rgba(75, 192, 192, 0.2)',
      borderColor: 'rgba(75, 192, 192, 1)',
      borderWidth: 1,
      yAxisID: 'y'
    }, {
      label: 'Visit Count',
      data: allStates.map(item => item.count),
      backgroundColor: 'rgba(255, 99, 132, 0.2)',
      borderColor: 'rgba(255, 99, 132, 1)',
      borderWidth: 1,
      yAxisID: 'y1'
    }]
  };

  // Chart options
  const timelineChartOptions = {
    responsive: true,
    scales: {
      y: {
        beginAtZero: true,
        max: 1.0
      }
    },
    plugins: {
      tooltip: {
        callbacks: {
          label: function(context) {
            const index = context.dataIndex;
            const item = noveltyData.timeline[index];
            return [
              `Novelty: ${formatNoveltyScore(item.novelty_score)}`,
              `Command: ${item.command || 'Unknown command'}`,
              `Success: ${item.success ? 'Yes' : 'No'}`
            ];
          }
        }
      }
    }
  };

  const distributionChartOptions = {
    responsive: true,
    scales: {
      y: {
        beginAtZero: true,
        position: 'left',
        title: {
          display: true,
          text: 'Novelty Score'
        },
        max: 1.0
      },
      y1: {
        beginAtZero: true,
        position: 'right',
        title: {
          display: true,
          text: 'Visit Count'
        },
        grid: {
          drawOnChartArea: false
        }
      }
    }
  };

  // Fetch data on component mount and set up auto-refresh
  useEffect(() => {
    fetchNoveltyData();

    // Set up auto-refresh every 30 seconds
    const intervalId = setInterval(fetchNoveltyData, 30000);

    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []);

  return (
    <Container fluid className="py-4">
      <h1 className="mb-4">Curiosity Mechanism Insights</h1>
      <p className="lead mb-4">Explore how the student model's curiosity mechanism is working</p>
      
      {/* Statistics Cards */}
      <Row className="mb-4">
        <Col md={3}>
          <Card className="bg-primary text-white mb-3">
            <Card.Body>
              <Card.Title>Total States</Card.Title>
              <h2>{noveltyData.statistics.total_states}</h2>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="bg-success text-white mb-3">
            <Card.Body>
              <Card.Title>Unique States</Card.Title>
              <h2>{noveltyData.statistics.unique_states}</h2>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="bg-info text-white mb-3">
            <Card.Body>
              <Card.Title>Total Visits</Card.Title>
              <h2>{noveltyData.statistics.total_visits}</h2>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="bg-warning text-white mb-3">
            <Card.Body>
              <Card.Title>Avg. Visits Per State</Card.Title>
              <h2>{noveltyData.statistics.avg_visits_per_state.toFixed(2)}</h2>
            </Card.Body>
          </Card>
        </Col>
      </Row>
      
      {/* Novelty Timeline Chart */}
      <Row>
        <Col md={8}>
          <Card className="mb-4">
            <Card.Header>
              <h5>Novelty Score Timeline</h5>
            </Card.Header>
            <Card.Body>
              {noveltyData.timeline.length > 0 ? (
                <Line 
                  data={timelineChartData} 
                  options={timelineChartOptions}
                  ref={timelineChartRef}
                />
              ) : (
                <div className="text-center py-5">
                  <p>No timeline data available yet.</p>
                </div>
              )}
            </Card.Body>
          </Card>
        </Col>
        <Col md={4}>
          <Card className="mb-4">
            <Card.Header>
              <h5>Most Novel States</h5>
            </Card.Header>
            <Card.Body style={{ maxHeight: '400px', overflowY: 'auto' }}>
              <Table striped hover size="sm">
                <thead>
                  <tr>
                    <th>State</th>
                    <th>Visits</th>
                    <th>Novelty</th>
                  </tr>
                </thead>
                <tbody>
                  {noveltyData.mostNovel.map((state, index) => (
                    <tr key={index}>
                      <td title={state.state}>{state.state.substring(0, 8)}...</td>
                      <td>{state.count}</td>
                      <td className={getNoveltyClass(state.novelty_score)}>
                        {formatNoveltyScore(state.novelty_score)}
                      </td>
                    </tr>
                  ))}
                  {noveltyData.mostNovel.length === 0 && (
                    <tr>
                      <td colSpan="3" className="text-center">No data available</td>
                    </tr>
                  )}
                </tbody>
              </Table>
            </Card.Body>
          </Card>
        </Col>
      </Row>
      
      <Row>
        <Col md={6}>
          <Card className="mb-4">
            <Card.Header>
              <h5>Least Novel States</h5>
            </Card.Header>
            <Card.Body style={{ maxHeight: '400px', overflowY: 'auto' }}>
              <Table striped hover size="sm">
                <thead>
                  <tr>
                    <th>State</th>
                    <th>Visits</th>
                    <th>Novelty</th>
                  </tr>
                </thead>
                <tbody>
                  {noveltyData.leastNovel.map((state, index) => (
                    <tr key={index}>
                      <td title={state.state}>{state.state.substring(0, 8)}...</td>
                      <td>{state.count}</td>
                      <td className={getNoveltyClass(state.novelty_score)}>
                        {formatNoveltyScore(state.novelty_score)}
                      </td>
                    </tr>
                  ))}
                  {noveltyData.leastNovel.length === 0 && (
                    <tr>
                      <td colSpan="3" className="text-center">No data available</td>
                    </tr>
                  )}
                </tbody>
              </Table>
            </Card.Body>
          </Card>
        </Col>
        <Col md={6}>
          <Card className="mb-4">
            <Card.Header>
              <h5>Novelty Distribution</h5>
            </Card.Header>
            <Card.Body>
              {allStates.length > 0 ? (
                <Bar 
                  data={distributionChartData} 
                  options={distributionChartOptions}
                  ref={distributionChartRef}
                />
              ) : (
                <div className="text-center py-5">
                  <p>No distribution data available yet.</p>
                </div>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
      
      <Row>
        <Col md={12}>
          <Card className="mb-4">
            <Card.Header className="d-flex justify-content-between align-items-center">
              <h5>Raw Curiosity Logs</h5>
              <div>
                <Button variant="primary" size="sm" onClick={fetchNoveltyData} className="me-2">
                  Refresh
                </Button>
                <Button variant="secondary" size="sm" onClick={downloadLogs}>
                  Download
                </Button>
              </div>
            </Card.Header>
            <Card.Body>
              <pre 
                style={{
                  backgroundColor: '#1e1e1e',
                  color: '#dcdcdc',
                  fontFamily: 'monospace',
                  padding: '10px',
                  borderRadius: '5px',
                  height: '400px',
                  overflowY: 'auto',
                  whiteSpace: 'pre-wrap',
                  wordWrap: 'break-word'
                }}
              >
                {noveltyData.recentLogs.length > 0 ? (
                  noveltyData.recentLogs.join('\n')
                ) : (
                  'No logs available yet.'
                )}
              </pre>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default NoveltyInsights;
