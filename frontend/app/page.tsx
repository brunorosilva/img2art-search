import Navbar from './components/Navbar';
import GalleryBackground from './components/GalleryBackground';
import Link from 'next/link';

export default function Home() {
  return (
    <div className="relative min-h-screen bg-stone-950 overflow-hidden">
      {/* Moving Gallery Background */}
      <GalleryBackground />

      {/* Navbar */}
      <Navbar />

      {/* Hero Content */}
      <main className="relative z-10 flex min-h-screen flex-col items-center justify-center px-6 pointer-events-none">
        <div className="text-center max-w-4xl mx-auto">
          {/* Main Title - Elegant serif */}
          <h1 className="font-serif text-5xl md:text-7xl lg:text-8xl text-stone-100 mb-8 leading-tight tracking-wide">
            <span className="block italic">DISCOVER ART</span>
          </h1>

          {/* CTA Button - Simple, elegant */}
          <Link
            href="/search"
            className="pointer-events-auto inline-block mt-8"
          >
            <button className="px-10 py-4 border border-stone-400 text-stone-200 text-lg tracking-widest uppercase hover:bg-stone-100 hover:text-stone-900 transition-all duration-300">
              Start Your Art Journey
            </button>
          </Link>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2">
          <span className="text-stone-500 text-xs tracking-wide uppercase">Hover to reveal</span>
        </div>
      </main>
    </div>
  );
}
