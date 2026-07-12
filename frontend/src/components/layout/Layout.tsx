/**
 * Layout — Master page structure wrapper.
 *
 * Part of: Frontend layout components
 * Used by: App.tsx (wraps all page content)
 *
 * Provides the overall page skeleton: sticky Header at top, scrollable main
 * content area in the middle, and footer at bottom. The main content area is
 * constrained to max-w-6xl for comfortable reading on wide screens.
 */

import type { ReactNode } from 'react';

import { Header } from './Header';

interface LayoutProps {
  children: ReactNode;  // Page content (HomePage, AboutPage, etc.)
}

export function Layout({ children }: LayoutProps) {
  // TWEAK NOTE: Layout structure and spacing
  // - flex flex-col: vertical stacking (Header, main, footer)
  // - min-h-screen: ensures footer sticks to bottom even with little content
  // - flex-1 on main: makes content area grow to fill available space
  // - max-w-6xl: content width limit; adjust to 5xl or 7xl for different proportions
  // - px-4/py-10: horizontal/vertical padding; adjust for narrower/wider margins
  return (
    <div className="flex min-h-screen flex-col">
      <Header />

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-10">
        {children}
      </main>

      <footer className="border-t border-white/10">
        <div className="mx-auto w-full max-w-6xl px-4 py-6 text-xs text-white/40">
          NightHaze — Final Year B.Tech Project · FFA-Net (AAAI 2020)
        </div>
      </footer>
    </div>
  );
}
