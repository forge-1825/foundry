import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  FileText,
  Database,
  Server,
  Cpu,
  BookOpen,
  Code,
  ChevronRight,
  ChevronDown,
  ExternalLink
} from 'lucide-react';

const Help = () => {
  const [expandedSections, setExpandedSections] = useState({
    'content-extraction-enrichment': true
  });

  const toggleSection = (sectionId) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };

  return (
    <div className="p-6">
      <div className="bg-white rounded-lg shadow-card p-6">
        <div className="flex items-center mb-6">
          <BookOpen size={24} className="text-primary-600 mr-3" />
          <h2 className="text-2xl font-semibold">Help & Documentation</h2>
        </div>

        <p className="mb-6 text-secondary-700">
          This documentation provides an overview of each step in the model distillation pipeline,
          explaining its purpose, functionality, and how to use it effectively.
        </p>

        {/* Table of Contents */}
        <div className="mb-8 p-4 bg-secondary-50 rounded-lg">
          <h3 className="text-lg font-medium mb-3">Pipeline Steps</h3>
          <ul className="space-y-2">
            <li>
              <Link to="#content-extraction-enrichment" className="flex items-center text-primary-600 hover:text-primary-800">
                <span className="w-6 h-6 rounded-full bg-primary-100 flex items-center justify-center mr-2">
                  <span className="text-primary-700 text-xs font-semibold">1</span>
                </span>
                Content Extraction & Enrichment
              </Link>
            </li>
            <li>
              <Link to="#teacher-pair-generation" className="flex items-center text-secondary-600 hover:text-secondary-800">
                <span className="w-6 h-6 rounded-full bg-secondary-100 flex items-center justify-center mr-2">
                  <span className="text-secondary-700 text-xs font-semibold">2</span>
                </span>
                Teacher Pair Generation
              </Link>
            </li>
            <li>
              <Link to="#distillation-training" className="flex items-center text-secondary-600 hover:text-secondary-800">
                <span className="w-6 h-6 rounded-full bg-secondary-100 flex items-center justify-center mr-2">
                  <span className="text-secondary-700 text-xs font-semibold">3</span>
                </span>
                Distillation Training
              </Link>
            </li>
            <li>
              <Link to="#student-self-study" className="flex items-center text-secondary-600 hover:text-secondary-800">
                <span className="w-6 h-6 rounded-full bg-secondary-100 flex items-center justify-center mr-2">
                  <span className="text-secondary-700 text-xs font-semibold">4</span>
                </span>
                Student Self-Study
              </Link>
            </li>
            <li>
              <Link to="#model-merging" className="flex items-center text-secondary-600 hover:text-secondary-800">
                <span className="w-6 h-6 rounded-full bg-secondary-100 flex items-center justify-center mr-2">
                  <span className="text-secondary-700 text-xs font-semibold">5</span>
                </span>
                Model Merging
              </Link>
            </li>
            <li>
              <Link to="#model-evaluation" className="flex items-center text-secondary-600 hover:text-secondary-800">
                <span className="w-6 h-6 rounded-full bg-secondary-100 flex items-center justify-center mr-2">
                  <span className="text-secondary-700 text-xs font-semibold">6</span>
                </span>
                Model Evaluation
              </Link>
            </li>
          </ul>
        </div>

        {/* Content Extraction & Enrichment Section */}
        <div id="content-extraction-enrichment" className="mb-8 scroll-mt-16">
          <div
            className="flex items-center cursor-pointer"
            onClick={() => toggleSection('content-extraction-enrichment')}
          >
            {expandedSections['content-extraction-enrichment'] ? (
              <ChevronDown size={20} className="text-primary-600 mr-2" />
            ) : (
              <ChevronRight size={20} className="text-primary-600 mr-2" />
            )}
            <h3 className="text-xl font-semibold flex items-center">
              <FileText size={20} className="text-primary-600 mr-2" />
              Content Extraction & Enrichment
            </h3>
          </div>

          {expandedSections['content-extraction-enrichment'] && (
            <div className="mt-4 pl-8">
              <div className="mb-4">
                <h4 className="text-lg font-medium mb-2">Overview</h4>
                <p className="text-secondary-700">
                  The Content Extraction & Enrichment step combines two essential processes: extracting raw content from various sources and enriching it with additional metadata and insights. This step prepares the data for the subsequent stages of the distillation pipeline.
                </p>
              </div>

              <div className="mb-4">
                <h4 className="text-lg font-medium mb-2">Key Features</h4>
                <ul className="list-disc pl-5 space-y-2 text-secondary-700">
                  <li>
                    <strong>Multiple Input Sources:</strong> Extract content from URLs, local folders, or Docker containers
                  </li>
                  <li>
                    <strong>PDF Generation:</strong> Convert web pages to PDF format for preservation
                  </li>
                  <li>
                    <strong>Link Extraction:</strong> Automatically extract and process links from web pages
                  </li>
                  <li>
                    <strong>Entity Extraction:</strong> Identify named entities such as people, organizations, and locations
                  </li>
                  <li>
                    <strong>Summarization:</strong> Generate concise summaries of the content using advanced NLP
                  </li>
                  <li>
                    <strong>Keyword Extraction:</strong> Identify important keywords and concepts
                  </li>
                  <li>
                    <strong>GPU Acceleration:</strong> Utilize GPU for faster processing of large datasets
                  </li>
                </ul>
              </div>

              <div className="mb-4">
                <h4 className="text-lg font-medium mb-2">How It Works</h4>
                <ol className="list-decimal pl-5 space-y-2 text-secondary-700">
                  <li>
                    <strong>Content Extraction:</strong> The system extracts raw content from the specified source (URL, local folder, or Docker container)
                  </li>
                  <li>
                    <strong>PDF Generation:</strong> For web content, the system generates PDF versions for preservation
                  </li>
                  <li>
                    <strong>Text Cleaning:</strong> The extracted text is cleaned and normalized to remove noise
                  </li>
                  <li>
                    <strong>Entity Extraction:</strong> Named entities are identified and categorized
                  </li>
                  <li>
                    <strong>Summarization:</strong> The content is summarized using advanced NLP techniques
                  </li>
                  <li>
                    <strong>Keyword Extraction:</strong> Important keywords and concepts are extracted
                  </li>
                  <li>
                    <strong>JSON Output:</strong> All extracted and enriched data is saved to a structured JSON file
                  </li>
                </ol>
              </div>

              <div className="mb-4">
                <h4 className="text-lg font-medium mb-2">Configuration Options</h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-secondary-200">
                    <thead className="bg-secondary-50">
                      <tr>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">Option</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">Description</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">Default</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-secondary-200">
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-secondary-900">URL</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-700">Web page URL to extract content from</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-700">None</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-secondary-900">Source Folder</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-700">Local folder containing files to process</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-700">None</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-secondary-900">Docker Folder</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-700">Folder path inside Docker container</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-700">/data</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-secondary-900">Output Directory</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-700">Directory to save extracted and enriched data</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-700">Output</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-secondary-900">Extract Links</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-700">Whether to extract and process links from web pages</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-700">False</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-secondary-900">Enable Enrichment</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-700">Whether to perform data enrichment after extraction</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-700">True</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-secondary-900">Use GPU</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-700">Whether to use GPU acceleration for processing</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-700">True</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="mb-4">
                <h4 className="text-lg font-medium mb-2">Tips for Best Results</h4>
                <ul className="list-disc pl-5 space-y-2 text-secondary-700">
                  <li>
                    <strong>Choose the Right Input Source:</strong> URLs work best for web content, while local folders are better for existing documents
                  </li>
                  <li>
                    <strong>Enable Link Extraction:</strong> For web content, enabling link extraction can provide more comprehensive results
                  </li>
                  <li>
                    <strong>Use GPU Acceleration:</strong> For large datasets, GPU acceleration can significantly speed up processing
                  </li>
                  <li>
                    <strong>Check Output Directory:</strong> Ensure the output directory exists and has sufficient space
                  </li>
                  <li>
                    <strong>Monitor System Resources:</strong> Keep an eye on CPU, memory, and GPU usage during processing
                  </li>
                </ul>
              </div>

              <div className="mt-6">
                <Link to="/scripts/content-extraction-enrichment" className="btn btn-primary flex items-center w-fit">
                  <FileText size={16} className="mr-2" />
                  Go to Content Extraction & Enrichment
                </Link>
              </div>
            </div>
          )}
        </div>

        {/* Teacher Pair Generation Section (Placeholder) */}
        <div id="teacher-pair-generation" className="mb-8 scroll-mt-16">
          <div
            className="flex items-center cursor-pointer"
            onClick={() => toggleSection('teacher-pair-generation')}
          >
            {expandedSections['teacher-pair-generation'] ? (
              <ChevronDown size={20} className="text-primary-600 mr-2" />
            ) : (
              <ChevronRight size={20} className="text-primary-600 mr-2" />
            )}
            <h3 className="text-xl font-semibold flex items-center">
              <Server size={20} className="text-primary-600 mr-2" />
              Teacher Pair Generation
            </h3>
          </div>

          {expandedSections['teacher-pair-generation'] && (
            <div className="mt-4 pl-8">
              <p className="text-secondary-500 italic">
                Documentation for this section is coming soon.
              </p>
            </div>
          )}
        </div>

        {/* Additional Pipeline Steps (Placeholders) */}
        <div id="distillation-training" className="mb-8 scroll-mt-16">
          <div
            className="flex items-center cursor-pointer"
            onClick={() => toggleSection('distillation-training')}
          >
            {expandedSections['distillation-training'] ? (
              <ChevronDown size={20} className="text-primary-600 mr-2" />
            ) : (
              <ChevronRight size={20} className="text-primary-600 mr-2" />
            )}
            <h3 className="text-xl font-semibold flex items-center">
              <Cpu size={20} className="text-primary-600 mr-2" />
              Distillation Training
            </h3>
          </div>

          {expandedSections['distillation-training'] && (
            <div className="mt-4 pl-8">
              <p className="text-secondary-500 italic">
                Documentation for this section is coming soon.
              </p>
            </div>
          )}
        </div>

        <div id="student-self-study" className="mb-8 scroll-mt-16">
          <div
            className="flex items-center cursor-pointer"
            onClick={() => toggleSection('student-self-study')}
          >
            {expandedSections['student-self-study'] ? (
              <ChevronDown size={20} className="text-primary-600 mr-2" />
            ) : (
              <ChevronRight size={20} className="text-primary-600 mr-2" />
            )}
            <h3 className="text-xl font-semibold flex items-center">
              <BookOpen size={20} className="text-primary-600 mr-2" />
              Student Self-Study
            </h3>
          </div>

          {expandedSections['student-self-study'] && (
            <div className="mt-4 pl-8">
              <p className="text-secondary-500 italic">
                Documentation for this section is coming soon.
              </p>
            </div>
          )}
        </div>

        <div id="model-merging" className="mb-8 scroll-mt-16">
          <div
            className="flex items-center cursor-pointer"
            onClick={() => toggleSection('model-merging')}
          >
            {expandedSections['model-merging'] ? (
              <ChevronDown size={20} className="text-primary-600 mr-2" />
            ) : (
              <ChevronRight size={20} className="text-primary-600 mr-2" />
            )}
            <h3 className="text-xl font-semibold flex items-center">
              <Database size={20} className="text-primary-600 mr-2" />
              Model Merging
            </h3>
          </div>

          {expandedSections['model-merging'] && (
            <div className="mt-4 pl-8">
              <p className="text-secondary-500 italic">
                Documentation for this section is coming soon.
              </p>
            </div>
          )}
        </div>

        <div id="model-evaluation" className="mb-8 scroll-mt-16">
          <div
            className="flex items-center cursor-pointer"
            onClick={() => toggleSection('model-evaluation')}
          >
            {expandedSections['model-evaluation'] ? (
              <ChevronDown size={20} className="text-primary-600 mr-2" />
            ) : (
              <ChevronRight size={20} className="text-primary-600 mr-2" />
            )}
            <h3 className="text-xl font-semibold flex items-center">
              <Code size={20} className="text-primary-600 mr-2" />
              Model Evaluation
            </h3>
          </div>

          {expandedSections['model-evaluation'] && (
            <div className="mt-4 pl-8">
              <p className="text-secondary-500 italic">
                Documentation for this section is coming soon.
              </p>
            </div>
          )}
        </div>

        {/* Additional Resources */}
        <div className="mt-12 p-4 bg-secondary-50 rounded-lg">
          <h3 className="text-lg font-medium mb-3">Additional Resources</h3>
          <ul className="space-y-2">
            <li>
              <a
                href="https://github.com/microsoft/phi-3"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center text-primary-600 hover:text-primary-800"
              >
                <ExternalLink size={16} className="mr-2" />
                Microsoft Phi-3 Model Documentation
              </a>
            </li>
            <li>
              <a
                href="https://huggingface.co/docs/transformers/model_doc/phi"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center text-primary-600 hover:text-primary-800"
              >
                <ExternalLink size={16} className="mr-2" />
                Hugging Face Phi Model Documentation
              </a>
            </li>
            <li>
              <a
                href="https://github.com/vllm-project/vllm"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center text-primary-600 hover:text-primary-800"
              >
                <ExternalLink size={16} className="mr-2" />
                vLLM Documentation
              </a>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Help;
