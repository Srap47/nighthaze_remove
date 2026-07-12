/**
 * Card — Glass-effect container for content.
 *
 * Part of: Frontend UI primitives
 * Used by: HomePage (feature cards, error card), MetricsDisplay (metric cards)
 *
 * A reusable wrapper applying a frosted-glass aesthetic (semi-transparent white
 * background with backdrop blur). Provides visual separation and grouping of content.
 */

import type { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
}

export function Card({ children, className = '' }: CardProps) {
  // TWEAK NOTE: Card glass effect styling
  // Adjust these to change the card appearance:
  // - rounded-2xl: border radius (use rounded-lg for less rounded, rounded-3xl for more)
  // - border-white/10: border opacity (higher % = more visible border)
  // - bg-white/5: background opacity (higher % = more opaque card, less transparent)
  // - backdrop-blur-sm: blur strength (use backdrop-blur for default, backdrop-blur-md for more blur)
  return (
    <div
      className={`rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm ${className}`}
    >
      {children}
    </div>
  );
}
