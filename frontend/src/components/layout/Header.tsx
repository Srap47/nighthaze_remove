import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { MoonStar } from 'lucide-react';

import { getHealth } from '../../services/api';

type HealthState = 'loading' | 'ready' | 'missing' | 'offline';

const INDICATORS: Record<HealthState, { dot: string; label: string }> = {
  loading: { dot: 'bg-white/30', label: 'Checking…' },
  ready: { dot: 'bg-emerald-400', label: 'Model ready' },
  missing: { dot: 'bg-amber-400', label: 'Model missing' },
  offline: { dot: 'bg-red-400', label: 'API offline' },
};

function navLinkClass({ isActive }: { isActive: boolean }): string {
  return `text-sm transition ${
    isActive ? 'text-white' : 'text-white/50 hover:text-white/80'
  }`;
}

export function Header() {
  const [health, setHealth] = useState<HealthState>('loading');

  useEffect(() => {
    let cancelled = false;

    getHealth()
      .then((res) => {
        if (!cancelled) setHealth(res.model_loaded ? 'ready' : 'missing');
      })
      .catch(() => {
        if (!cancelled) setHealth('offline');
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const indicator = INDICATORS[health];

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
