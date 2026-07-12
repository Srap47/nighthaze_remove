/**
 * ImageComparison — Draggable before/after slider.
 *
 * Part of: Frontend viewer components
 * Used by: HomePage (in "done" state tab, shows hazy vs dehazed side-by-side)
 *
 * Allows users to drag a vertical divider left/right to compare the original
 * hazy image with the dehazed result. Implemented from scratch using Pointer
 * Events API (no external library) so mouse, touch, and stylus input work
 * with a single code path. The "before" (hazy) image sits on top and is
 * clipped via CSS clipPath; dragging adjusts the clipped region.
 */

import { useCallback, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface ImageComparisonProps {
  beforeSrc: string; // data:image/png;base64,...  (hazy original)
  afterSrc: string;  // data:image/png;base64,...  (dehazed result)
  className?: string;
}

export function ImageComparison({
  beforeSrc,
  afterSrc,
  className = '',
}: ImageComparisonProps) {
  // Slider position as percentage (0-100, where 0 = full "after", 100 = full "before")
  const [sliderPos, setSliderPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);  // Track whether user is actively dragging

  // Convert absolute clientX position to percentage slider position (0-100)
  const updateFromClientX = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0) return;
    // Calculate position as percentage from left edge
    const pct = ((clientX - rect.left) / rect.width) * 100;
    // Clamp to 0-100 range
    setSliderPos(Math.min(100, Math.max(0, pct)));
  }, []);

  // Pointer events (mouse, touch, stylus) — start drag
  // setPointerCapture ensures we track movement even outside the container
  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      isDragging.current = true;
      event.currentTarget.setPointerCapture(event.pointerId);
      updateFromClientX(event.clientX);
    },
    [updateFromClientX],
  );

  // Pointer events — update position during drag
  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (!isDragging.current) return;  // Ignore if not actively dragging
      updateFromClientX(event.clientX);
    },
    [updateFromClientX],
  );

  // Pointer events — end drag
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
      // TWEAK NOTE: Slider interaction styling
      // - cursor-ew-resize: shows east-west resize cursor to indicate draggable divider
      // - select-none touch-none: prevent text selection and default browser touch behavior
      // - rounded-2xl: corner radius (adjust to match card borders elsewhere)
      className={`relative select-none touch-none overflow-hidden rounded-2xl cursor-ew-resize ${className}`}
    >
      {/* Base layer: dehazed (after) image fills the container and defines aspect ratio. */}
      <img
        src={afterSrc}
        alt="Dehazed"
        draggable={false}
        className="block h-auto w-full"
      />

      {/* Overlay layer: hazy (before) image sits on top.
          clipPath: inset(0 right% 0 0) clips from the right, progressively revealing
          the before image as sliderPos increases. At 0%, the before image is completely
          hidden (right=100%); at 100%, it's fully visible (right=0%).
      */}
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

      {/* Vertical white divider line and circular drag handle.
          Positioned at sliderPos%, so it moves as the user drags.
      */}
      <div
        className="absolute bottom-0 top-0 w-0.5 bg-white"
        style={{ left: `${sliderPos}%` }}
      >
        {/* Drag handle: circular button with chevron icons. */}
        <div className="absolute left-1/2 top-1/2 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white shadow-lg">
          <ChevronLeft className="h-4 w-4 text-black/70" />
          <ChevronRight className="-ml-1 h-4 w-4 text-black/70" />
        </div>
      </div>

      {/* Corner labels: "Hazy" (left) and "Dehazed" (right) */}
      <span className="absolute left-2 top-2 rounded bg-black/60 px-2 py-1 text-xs">
        Hazy
      </span>
      <span className="absolute right-2 top-2 rounded bg-black/60 px-2 py-1 text-xs">
        Dehazed
      </span>
    </div>
  );
}
