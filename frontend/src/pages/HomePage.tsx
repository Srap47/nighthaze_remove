/**
 * HomePage: Main application page orchestrating the full user workflow.
 *
 * Renders a state machine driven by useDehazeAPI hook:
 * - idle: hero + upload prompt + features
 * - uploading: progress bar
 * - processing: spinner + pipeline stage breakdown
 * - done: before/after comparison + metrics + download
 * - error: error message + retry button
 *
 * Displays three visualization tabs in results:
 * - Result: side-by-side before/after comparison
 * - Transmission Map: haze density visualization
 * - Glow Mask: detected light source regions
 */

import { useState } from 'react';
import { AlertTriangle, Atom, Brain, Gauge } from 'lucide-react';

import { PipelineProgress } from '../components/pipeline/PipelineProgress';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { Spinner } from '../components/ui/Spinner';
import { DropZone } from '../components/uploader/DropZone';
import { ImageComparison } from '../components/viewer/ImageComparison';
import { MetricsDisplay } from '../components/viewer/MetricsDisplay';
import { useDehazeAPI } from '../hooks/useDehazeAPI';

/** Visualization tabs in the results panel. */
type Tab = 'result' | 'transmission' | 'glow';

const TABS: { id: Tab; label: string }[] = [
  { id: 'result', label: 'Result' },
  { id: 'transmission', label: 'Transmission Map' },
  { id: 'glow', label: 'Glow Mask' },
];

const FEATURES = [
  {
    icon: Brain,
    title: 'Attention-Based DL',
    text: 'FFA-Net channel & pixel attention',
  },
  {
    icon: Atom,
    title: 'Physics Model',
    text: 'Atmospheric scattering inversion',
  },
  {
    icon: Gauge,
    title: 'Real Metrics',
    text: 'PSNR, SSIM, BRISQUE, NIQE',
  },
];

export function HomePage() {
  const {
    status,
    result,
    error,
    uploadProgress,
    processImage,
    loadDemo,
    reset,
  } = useDehazeAPI();
  const [activeTab, setActiveTab] = useState<Tab>('result');

  const downloadResult = () => {
    if (!result) return;
    const link = document.createElement('a');
    link.href = result.dehazed_image_b64;
    link.download = 'nighthaze_result.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (status === 'uploading') {
    return (
      <div className="flex flex-col items-center justify-center gap-6 py-24">
        <Spinner size="lg" />
        <div className="h-2 w-64 overflow-hidden rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-gradient-to-r from-primary to-accent transition-all duration-200"
            style={{ width: `${uploadProgress}%` }}
          />
        </div>
        <p className="text-sm text-white/60">Uploading… {uploadProgress}%</p>
      </div>
    );
  }

  if (status === 'processing') {
    return (
      <div className="flex flex-col items-center justify-center gap-8 py-20">
        <Spinner size="lg" />
        <PipelineProgress />
        <p className="text-sm text-white/60">Removing haze from your image…</p>
      </div>
    );
  }

  if (status === 'error') {
    return (
      <div className="mx-auto max-w-lg py-20">
        <Card className="flex flex-col items-center gap-4 text-center">
          <AlertTriangle className="h-10 w-10 text-red-400" />
          <h2 className="text-lg font-semibold">Something went wrong</h2>
          <p className="text-sm text-white/60">{error}</p>
          <Button onClick={reset}>Try Again</Button>
        </Card>
      </div>
    );
  }

  if (status === 'done' && result) {
    return (
      <div className="space-y-8">
        {/* View switcher */}
        <div className="flex flex-wrap gap-2">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                activeTab === tab.id
                  ? 'bg-primary text-white'
                  : 'bg-white/5 text-white/60 hover:bg-white/10'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {activeTab === 'result' && (
          <ImageComparison
            beforeSrc={result.original_image_b64}
            afterSrc={result.dehazed_image_b64}
          />
        )}

        {activeTab === 'transmission' && (
          <Card>
            <img
              src={result.transmission_map_b64}
              alt="Transmission map"
              className="w-full rounded-xl"
            />
          </Card>
        )}

        {activeTab === 'glow' && (
          <Card>
            <img
              src={result.glow_mask_b64}
              alt="Glow mask"
              className="w-full rounded-xl"
            />
          </Card>
        )}

        <MetricsDisplay
          metrics={result.metrics}
          stages={result.pipeline_stages}
        />

        <div className="flex flex-wrap justify-center gap-3">
          <Button onClick={downloadResult}>Download Result</Button>
          <Button variant="ghost" onClick={reset}>
            Process Another Image
          </Button>
        </div>
      </div>
    );
  }

  // 'idle'
  return (
    <div className="space-y-14">
      <section className="mx-auto max-w-3xl text-center">
        <h1 className="text-4xl font-extrabold tracking-tight md:text-5xl">
          Remove Haze from{' '}
          <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            Night Images
          </span>
        </h1>
        <p className="mt-5 text-white/60">
          Physics-based pipeline with an FFA-Net deep learning backbone — built
          for the unique challenges of nighttime photography.
        </p>
      </section>

      <section className="mx-auto max-w-2xl space-y-6">
        <DropZone onFileSelect={processImage} />

        <div className="flex items-center gap-4">
          <span className="h-px flex-1 bg-white/10" />
          <span className="text-xs uppercase tracking-wider text-white/30">
            or
          </span>
          <span className="h-px flex-1 bg-white/10" />
        </div>

        <div className="flex justify-center">
          <Button variant="ghost" onClick={loadDemo}>
            Try with Demo Image
          </Button>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        {FEATURES.map((feature) => {
          const Icon = feature.icon;
          return (
            <Card key={feature.title} className="text-center">
              <Icon className="mx-auto h-6 w-6 text-accent" />
              <h3 className="mt-3 font-semibold">{feature.title}</h3>
              <p className="mt-1 text-sm text-white/50">{feature.text}</p>
            </Card>
          );
        })}
      </section>
    </div>
  );
}
