import React from 'react';
import { Nav } from 'react-bootstrap';
import { Link, useLocation } from 'react-router-dom';

const Sidebar = () => {
  const location = useLocation();
  
  // Define sidebar items
  const sidebarItems = [
    { path: '/', label: 'Home', icon: 'bi-house-door' },
    { path: '/content-extraction', label: 'Content Extraction & Enrichment', icon: 'bi-file-earmark-text' },
    { path: '/teacher-pair-generation', label: 'Teacher Pair Generation', icon: 'bi-people' },
    { path: '/distillation', label: 'Distillation Training', icon: 'bi-lightning' },
    { path: '/model-merging', label: 'Model Merging', icon: 'bi-box' },
    { path: '/student-self-study', label: 'Student Self-Study', icon: 'bi-book' },
    { path: '/model-evaluation', label: 'Model Evaluation', icon: 'bi-graph-up' },
    { divider: true },
    { path: '/settings', label: 'Settings', icon: 'bi-gear' },
  ];
  
  return (
    <div className="sidebar bg-light border-end">
      <div className="sidebar-header p-3 border-bottom">
        <h5 className="mb-0">Distillation Pipeline</h5>
      </div>
      <Nav className="flex-column p-3">
        {sidebarItems.map((item, index) => (
          item.divider ? (
            <hr key={`divider-${index}`} className="my-3" />
          ) : (
            <Nav.Item key={item.path}>
              <Nav.Link 
                as={Link} 
                to={item.path} 
                className={`d-flex align-items-center ${location.pathname === item.path ? 'active' : ''}`}
              >
                <i className={`bi ${item.icon} me-2`}></i>
                {item.label}
              </Nav.Link>
            </Nav.Item>
          )
        ))}
      </Nav>
    </div>
  );
};

export default Sidebar;
