// Before/after drag slider — implemented from scratch (no external library).
// Uses Pointer Events + setPointerCapture, so mouse, touch and pen all work
// with a single code path.

import { useCallback, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface ImageComparisonProps {
  beforeSrc: string; // hazy original
  afterSrc: string; // dehazed
  className?: string;
}

export function ImageComparison({
  beforeSrc,
  afterSrc,
  className = '',
}: ImageComparisonProps) {
  const [sliderPos, setSliderPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  const updateFromClientX = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0) return;
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setSliderPos(Math.min(100, Math.max(0, pct)));
  }, []);

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      isDragging.current = true;
      event.currentTarget.setPointerCapture(event.pointerId);
      updateFromClientX(event.clientX);
    },
    [updateFromClientX],
  );

  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (!isDragging.current) return;
      updateFromClientX(event.clientX);
    },
    [updateFromClientX],
  );

  const handlePointerUp = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      isDragging.current = false;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
    },
    [],
  );

  return (
    <div
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      className={`relative select-none touch-none overflow-hidden rounded-2xl cursor-ew-resize ${className}`}
    >
      {/* Dehazed (after) fills the container and defines its height. */}
      <img
        src={afterSrc}
        alt="Dehazed"
        draggable={false}
        className="block h-auto w-full"
      />

      {/* Hazy (before) sits on top, clipped to the slider position. */}
      <div
        className="absolute inset-0"
        style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
      >
        <img
          src={beforeSrc}
          alt="Hazy original"
          draggable={false}
          className="block h-auto w-full"
        />
      </div>

      {/* Divider + drag handle. */}
      <div
        className="absolute bottom-0 top-0 w-0.5 bg-white"
        style={{ left: `${sliderPos}%` }}
      >
        <div className="absolute left-1/2 top-1/2 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white shadow-lg">
          <ChevronLeft className="h-4 w-4 text-black/70" />
          <ChevronRight className="-ml-1 h-4 w-4 text-black/70" />
        </div>
      </div>

      <span className="absolute left-2 top-2 rounded bg-black/60 px-2 py-1 text-xs">
        Hazy
      </span>
      <span className="absolute right-2 top-2 rounded bg-black/60 px-2 py-1 text-xs">
        Dehazed
      </span>
    </div>
  );
}
