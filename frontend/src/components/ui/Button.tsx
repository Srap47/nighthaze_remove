/**
 * Button — Reusable button component with theme-aware variants.
 *
 * Part of: Frontend UI primitives
 * Used by: HomePage, AboutPage, DropZone (all interactive buttons)
 *
 * Supports two visual variants (primary gradient, ghost outline) and two sizes
 * (md/lg). Inherits all standard HTML button attributes. Automatically handles
 * disabled state (opacity + no hover effect).
 */

import type { ButtonHTMLAttributes, ReactNode } from 'react';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'ghost';
  size?: 'md' | 'lg';
  children: ReactNode;
  className?: string;
}

// TWEAK NOTE: Button variant styling
// Modify these Tailwind classes to change button appearance:
// - primary: gradient from config.theme.primary to accent (currently purple to cyan)
// - ghost: outline style with minimal background (currently white/15 border, white/5 bg)
const VARIANTS: Record<NonNullable<ButtonProps['variant']>, string> = {
  primary:
    'bg-gradient-to-r from-primary to-accent text-white hover:brightness-110',
  ghost: 'border border-white/15 bg-white/5 text-white hover:bg-white/10',
};

// TWEAK NOTE: Button sizes
// Adjust padding and font size to change button proportions
const SIZES: Record<NonNullable<ButtonProps['size']>, string> = {
  md: 'px-4 py-2 text-sm',
  lg: 'px-6 py-3 text-base',
};

export function Button({
  variant = 'primary',
  size = 'md',
  className = '',
  children,
  ...rest
}: ButtonProps) {
  const base =
    'inline-flex items-center justify-center gap-2 rounded-xl font-semibold transition ' +
    'disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:brightness-100';

  return (
    <button
      className={`${base} ${VARIANTS[variant]} ${SIZES[size]} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}
