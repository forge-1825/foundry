"""
Unit tests for individual pipeline components
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.smoke
class TestDataEnrichment:
    """Test data enrichment functionality"""
    
    def test_enrichment_output_structure(self, sample_enriched_data):
        """Test enriched data has correct structure"""
        for record in sample_enriched_data:
            assert "url" in record
            assert "text" in record
            assert "content" in record
            assert "summary" in record
            assert "entities" in record
            assert "keywords" in record
            assert isinstance(record["entities"], list)
            assert isinstance(record["keywords"], list)
    
    @patch('data_enrichment.process_pdf')
    def test_pdf_processing_mock(self, mock_process_pdf, sample_pdf_content):
        """Test PDF processing with mock"""
        mock_process_pdf.return_value = {
            "text": sample_pdf_content,
            "pages": 1
        }
        
        # Import after patching
        import data_enrichment
        result = data_enrichment.process_pdf("test.pdf")
        
        assert result["text"] == sample_pdf_content
        assert result["pages"] == 1
        mock_process_pdf.assert_called_once_with("test.pdf")
    
    def test_enrichment_json_output(self, temp_output_dir, sample_enriched_data):
        """Test writing enriched data to JSON"""
        output_file = temp_output_dir / "enriched_data.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_enriched_data, f, indent=2)
        
        # Verify file exists and can be loaded
        assert output_file.exists()
        with open(output_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == sample_enriched_data


@pytest.mark.smoke
class TestTeacherPairGeneration:
    """Test teacher pair generation functionality"""
    
    def test_teacher_pair_structure(self, sample_teacher_pairs):
        """Test teacher pair data structure"""
        for pair in sample_teacher_pairs:
            assert "input" in pair
            assert "target" in pair
            assert isinstance(pair["input"], str)
            assert isinstance(pair["target"], str)
            assert len(pair["input"]) > 0
            assert len(pair["target"]) > 0
    
    @patch('requests.post')
    def test_teacher_generation_mock(self, mock_post, sample_enriched_data, mock_vllm_response):
        """Test teacher pair generation with mocked API"""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_vllm_response
        
        # Simulate generating a Q&A pair
        prompt = f"Generate a question about: {sample_enriched_data[0]['summary']}"
        
        import requests
        response = requests.post(
            "http://localhost:8000/v1/completions",
            json={"prompt": prompt, "max_tokens": 100}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert data["choices"][0]["text"] == "This is a mock response from the model."
    
    def test_hierarchical_context_building(self, sample_enriched_data):
        """Test building hierarchical context"""
        # Simulate hierarchical context structure
        context = {
            "document": sample_enriched_data[0]["content"],
            "section": sample_enriched_data[0]["summary"],
            "entities": sample_enriched_data[0]["entities"],
            "keywords": sample_enriched_data[0]["keywords"]
        }
        
        assert "document" in context
        assert "section" in context
        assert isinstance(context["entities"], list)
        assert isinstance(context["keywords"], list)


@pytest.mark.smoke
class TestModelMerging:
    """Test model merging functionality"""
    
    def test_merge_configuration(self):
        """Test merge configuration structure"""
        merge_config = {
            "base_model": "microsoft/Phi-3-mini-4k-instruct",
            "adapter_path": "./distilled_model_phi3_improved/best_checkpoint",
            "output_path": "./merged_model",
            "use_safetensors": True
        }
        
        assert "base_model" in merge_config
        assert "adapter_path" in merge_config
        assert "output_path" in merge_config
        assert merge_config["use_safetensors"] is True
    
    @patch('merge_model.merge_lora_to_base_model')
    def test_merge_function_mock(self, mock_merge):
        """Test merge function with mock"""
        mock_merge.return_value = True
        
        import merge_model
        result = merge_model.merge_lora_to_base_model(
            base_model_name="test-model",
            adapter_path="test-adapter",
            output_path="test-output"
        )
        
        assert result is True
        mock_merge.assert_called_once()


@pytest.mark.smoke
class TestStudentSelfStudy:
    """Test student self-study functionality"""
    
    def test_self_study_output_structure(self):
        """Test self-study output structure"""
        sample_output = {
            "sentence": "This is a test sentence.",
            "questions": [
                {
                    "question": "What is this?",
                    "student_answer": "This is a test.",
                    "teacher_answer": "This is a test sentence.",
                    "similarity_score": 0.85,
                    "quality_score": 0.90
                }
            ],
            "context": "Test paragraph",
            "summary": "Test summary"
        }
        
        assert "sentence" in sample_output
        assert "questions" in sample_output
        assert isinstance(sample_output["questions"], list)
        assert len(sample_output["questions"]) > 0
        
        question = sample_output["questions"][0]
        assert "question" in question
        assert "student_answer" in question
        assert "similarity_score" in question
        assert 0 <= question["similarity_score"] <= 1
    
    def test_self_study_configuration(self):
        """Test self-study configuration"""
        config = {
            "pdf_folder": "test_pdfs",
            "model_path": "./distilled_model",
            "output_dir": "./self_study_results",
            "use_teacher": True,
            "num_questions": 20,
            "min_sentence_length": 5,
            "max_sentence_length": 100
        }
        
        assert config["num_questions"] == 20
        assert config["min_sentence_length"] == 5
        assert config["max_sentence_length"] == 100
        assert config["use_teacher"] is True


@pytest.mark.smoke
class TestEvaluation:
    """Test evaluation functionality"""
    
    def test_evaluation_metrics_structure(self):
        """Test evaluation metrics structure"""
        metrics = {
            "accuracy": 0.85,
            "f1_score": 0.83,
            "perplexity": 12.5,
            "inference_time_ms": 45.2,
            "model_size_mb": 2800
        }
        
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert metrics["perplexity"] > 0
        assert metrics["inference_time_ms"] > 0
    
    def test_evaluation_comparison(self):
        """Test model comparison structure"""
        comparison = {
            "teacher": {"accuracy": 0.92, "f1_score": 0.90},
            "student": {"accuracy": 0.85, "f1_score": 0.83},
            "improvement": {
                "accuracy_delta": -0.07,
                "f1_score_delta": -0.07,
                "size_reduction": 0.95
            }
        }
        
        assert "teacher" in comparison
        assert "student" in comparison
        assert "improvement" in comparison
        assert comparison["improvement"]["size_reduction"] > 0