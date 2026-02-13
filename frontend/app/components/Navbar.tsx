'use client';

import Link from 'next/link';

export default function Navbar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass pointer-events-auto">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <span className="text-white text-sm font-bold">i2a</span>
            </div>
            <span className="text-white font-semibold text-lg tracking-tight">
              img2art
            </span>
          </Link>

          {/* Nav Links */}
          <div className="hidden md:flex items-center gap-8">
            <Link href="#how-it-works" className="text-white/70 hover:text-white transition-colors text-sm">
              How it works
            </Link>
            <Link href="#examples" className="text-white/70 hover:text-white transition-colors text-sm">
              Examples
            </Link>
            <Link href="https://github.com/brunorosilva/img2art-search" target="_blank" className="text-white/70 hover:text-white transition-colors text-sm">
              GitHub
            </Link>
          </div>

          {/* CTA */}
          <Link
            href="/search"
            className="glass-strong px-4 py-2 rounded-full text-white text-sm font-medium hover:bg-white/20 transition-all"
          >
            Launch App
          </Link>
        </div>
      </div>
    </nav>
  );
}
