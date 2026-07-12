/**
 * Header — Sticky navigation bar with health status indicator.
 *
 * Part of: Frontend layout components
 * Used by: Layout (appears at top of every page)
 *
 * Displays the app logo, navigation links (Home/About), and a colored dot
 * indicating whether the backend model is loaded. On mount, fetches the
 * health endpoint to determine model readiness.
 */

import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { MoonStar } from 'lucide-react';

import { getHealth } from '../../services/api';

type HealthState = 'loading' | 'ready' | 'missing' | 'offline';

// TWEAK NOTE: Health indicator appearance
// Modify colors and labels to change what users see for each state:
// - loading: gray, appears while health endpoint is being polled
// - ready: emerald (green), model loaded and ready to process
// - missing: amber (yellow), model weights were not found (graceful degradation)
// - offline: red, API is unreachable or health endpoint failed
const INDICATORS: Record<HealthState, { dot: string; label: string }> = {
  loading: { dot: 'bg-white/30', label: 'Checking…' },
  ready: { dot: 'bg-emerald-400', label: 'Model ready' },
  missing: { dot: 'bg-amber-400', label: 'Model missing' },
  offline: { dot: 'bg-red-400', label: 'API offline' },
};

// Helper function for react-router NavLink active state styling
function navLinkClass({ isActive }: { isActive: boolean }): string {
  return `text-sm transition ${
    isActive ? 'text-white' : 'text-white/50 hover:text-white/80'
  }`;
}

export function Header() {
  const [health, setHealth] = useState<HealthState>('loading');

  // Poll the health endpoint once on mount to determine backend readiness.
  // If the API is unreachable or the model fails to load, show the appropriate state.
  useEffect(() => {
    let cancelled = false;

    getHealth()
      .then((res) => {
        // Model successfully loaded and available
        if (!cancelled) setHealth(res.model_loaded ? 'ready' : 'missing');
      })
      .catch(() => {
        // API is unreachable or health check failed
        if (!cancelled) setHealth('offline');
      });

    // Cleanup function: if component unmounts before request completes, don't update state
    return () => {
      cancelled = true;
    };
  }, []);

  const indicator = INDICATORS[health];

  // TWEAK NOTE: Header sticky positioning
  // sticky top-0 z-50 keeps the header fixed at the top while scrolling.
  // Adjust z-50 if dropdowns/modals need to appear above; use z-40 if they should go below.
  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-night/80 backdrop-blur">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-3">
        <NavLink to="/" className="flex items-center gap-2">
          <MoonStar className="h-5 w-5 text-primary" />
          <span className="text-lg font-bold tracking-tight">
            Night<span className="text-primary">Haze</span>
          </span>
        </NavLink>

        <div className="flex items-center gap-6">
          <nav className="flex items-center gap-4">
            <NavLink to="/" end className={navLinkClass}>
              Home
            </NavLink>
            <NavLink to="/about" className={navLinkClass}>
              About
            </NavLink>
          </nav>

          <div className="flex items-center gap-1.5 text-xs text-white/50">
            <span className={`h-2 w-2 rounded-full ${indicator.dot}`} />
            <span>{indicator.label}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
