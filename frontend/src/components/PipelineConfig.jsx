import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Alert } from './ui/alert';

export function PipelineConfig({ step, onStart }) {
  const [params, setParams] = useState({
    batch_size: 32,
    learning_rate: 1e-4,
    num_epochs: 3,
    max_length: 512,
    lora_r: 64,
    lora_alpha: 128,
    lora_dropout: 0.1
  });

  const [error, setError] = useState(null);

  const handleParamChange = (key, value) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async () => {
    try {
      setError(null);
      await onStart(params);
    } catch (err) {
      setError(err.message);
    }
  };

  // Custom parameter fields based on step
  const getStepParams = () => {
    switch (step.id) {
      case 'distillation':
        return (
          <>
            <div className="space-y-2">
              <label>LoRA Rank</label>
              <Input 
                type="number"
                value={params.lora_r}
                onChange={e => handleParamChange('lora_r', parseInt(e.target.value))}
              />
            </div>
            <div className="space-y-2">
              <label>LoRA Alpha</label>
              <Input 
                type="number"
                value={params.lora_alpha}
                onChange={e => handleParamChange('lora_alpha', parseInt(e.target.value))}
              />
            </div>
            <div className="space-y-2">
              <label>LoRA Dropout</label>
              <Input 
                type="number"
                step="0.01"
                value={params.lora_dropout}
                onChange={e => handleParamChange('lora_dropout', parseFloat(e.target.value))}
              />
            </div>
          </>
        );
      default:
        return null;
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Configure {step.name}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="space-y-2">
            <label>Batch Size</label>
            <Input 
              type="number"
              value={params.batch_size}
              onChange={e => handleParamChange('batch_size', parseInt(e.target.value))}
            />
          </div>
          <div className="space-y-2">
            <label>Learning Rate</label>
            <Input 
              type="number"
              step="0.0001"
              value={params.learning_rate}
              onChange={e => handleParamChange('learning_rate', parseFloat(e.target.value))}
            />
          </div>
          <div className="space-y-2">
            <label>Number of Epochs</label>
            <Input 
              type="number"
              value={params.num_epochs}
              onChange={e => handleParamChange('num_epochs', parseInt(e.target.value))}
            />
          </div>
          <div className="space-y-2">
            <label>Max Length</label>
            <Input 
              type="number"
              value={params.max_length}
              onChange={e => handleParamChange('max_length', parseInt(e.target.value))}
            />
          </div>
          
          {getStepParams()}

          {error && (
            <Alert variant="destructive">
              {error}
            </Alert>
          )}

          <Button onClick={handleSubmit}>
            Start {step.name}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
