import type { ReactNode } from 'react';

import { Header } from './Header';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
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
