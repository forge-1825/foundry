import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import ScriptConfiguration from './pages/ScriptConfiguration';
import ContentExtractionEnrichment from './pages/ContentExtractionEnrichment';
import TeacherPairGeneration from './pages/TeacherPairGeneration';
import DistillationPhase from './pages/DistillationPhase';
import StudentSelfStudy from './pages/StudentSelfStudy';
import LogMonitor from './pages/LogMonitor';
import FileBrowser from './pages/FileBrowser';
import Results from './pages/Results';
import ModelInteraction from './pages/ModelInteraction';
import Help from './pages/Help';
import NoveltyInsights from './pages/NoveltyInsights';
import ModelEvaluationHub from './pages/ModelEvaluationHub';
import EvaluationResultsVisualization from './pages/EvaluationResultsVisualization';
import { SystemProvider } from './contexts/SystemContext';
import { ScriptProvider } from './contexts/ScriptContext';
import { LogProvider } from './contexts/LogContext';
import { FileProvider } from './contexts/FileContext';

function App() {
  return (
    <SystemProvider>
      <ScriptProvider>
        <LogProvider>
          <FileProvider>
            <Layout>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/scripts" element={<Navigate to="/scripts/content_extraction_enrichment" />} />
                {/* Redirect manual_extractor and data_enrichment to content_extraction_enrichment */}
                <Route path="/scripts/manual_extractor" element={<Navigate to="/scripts/content_extraction_enrichment" />} />
                <Route path="/scripts/data_enrichment" element={<Navigate to="/scripts/content_extraction_enrichment" />} />
                <Route path="/scripts/content_extraction_enrichment" element={<ContentExtractionEnrichment />} />
                <Route path="/scripts/teacher_pair_generation" element={<TeacherPairGeneration />} />
                <Route path="/scripts/distillation" element={<DistillationPhase />} />
                <Route path="/scripts/student_self_study" element={<StudentSelfStudy />} />
                <Route path="/scripts/merge_model" element={<ScriptConfiguration />} />
                <Route path="/scripts/evaluation" element={<ModelEvaluationHub />} />
                <Route path="/evaluation/results/:runId" element={<EvaluationResultsVisualization />} />
                <Route path="/scripts/:scriptId" element={<ScriptConfiguration />} />
                <Route path="/logs/:scriptId?" element={<LogMonitor />} />
                <Route path="/files/*" element={<FileBrowser />} />
                <Route path="/results" element={<Results />} />
                <Route path="/model-interaction" element={<ModelInteraction />} />
                <Route path="/novelty-insights" element={<NoveltyInsights />} />
                <Route path="/help" element={<Help />} />
              </Routes>
            </Layout>
          </FileProvider>
        </LogProvider>
      </ScriptProvider>
    </SystemProvider>
  );
}

export default App;
