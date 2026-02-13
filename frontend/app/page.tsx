import Navbar from './components/Navbar';
import GalleryBackground from './components/GalleryBackground';
import Link from 'next/link';

export default function Home() {
  return (
    <div className="relative min-h-screen bg-black">
      {/* Moving Gallery Background */}
      <GalleryBackground />

      {/* Glass Navbar */}
      <Navbar />

      {/* Hero Content - pointer-events-none on container, auto on interactive elements */}
      <main className="relative z-10 flex min-h-screen flex-col items-center justify-center px-6 pointer-events-none">
        <div className="text-center max-w-3xl mx-auto">
          {/* Tagline */}
          <p className="text-white/60 text-sm font-medium tracking-widest uppercase mb-4">
            Discover Art Through Your Images
          </p>

          {/* Main Title */}
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
            Find artwork that
            <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent">
              {' '}matches{' '}
            </span>
            your photos
          </h1>

          {/* Subtitle */}
          <p className="text-white/50 text-lg md:text-xl mb-12 max-w-xl mx-auto">
            Upload any image and discover visually similar masterpieces from 81,000+ artworks in the WikiArt collection.
          </p>

          {/* CTA Button */}
          <Link
            href="/search"
            className="group relative inline-flex items-center justify-center pointer-events-auto"
          >
            <div className="absolute -inset-1 bg-gradient-to-r from-purple-600 to-pink-600 rounded-full blur-lg opacity-70 group-hover:opacity-100 transition-opacity" />
            <button className="relative btn-glow glass-strong px-12 py-5 rounded-full text-white text-xl font-semibold hover:bg-white/20 transition-all duration-300 transform hover:scale-105">
              Try it
            </button>
          </Link>

          {/* Stats */}
          <div className="mt-16 flex items-center justify-center gap-12 text-center">
            <div>
              <p className="text-3xl font-bold text-white">81k+</p>
              <p className="text-white/40 text-sm">Artworks</p>
            </div>
            <div className="w-px h-12 bg-white/10" />
            <div>
              <p className="text-3xl font-bold text-white">ViT</p>
              <p className="text-white/40 text-sm">Powered</p>
            </div>
            <div className="w-px h-12 bg-white/10" />
            <div>
              <p className="text-3xl font-bold text-white">Free</p>
              <p className="text-white/40 text-sm">Open Source</p>
            </div>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2">
          <span className="text-white/30 text-xs">Hover cards to see matches</span>
          <div className="w-6 h-10 rounded-full border-2 border-white/20 flex items-start justify-center p-2">
            <div className="w-1.5 h-1.5 rounded-full bg-white/40 animate-bounce" />
          </div>
        </div>
      </main>
    </div>
  );
}
