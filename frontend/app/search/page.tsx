'use client';

import { useState, useCallback } from 'react';
import Link from 'next/link';
import Image from 'next/image';

// Hugging Face Space API URL
const API_BASE_URL = 'https://chicelli-img2art-search.hf.space';

interface ArtworkResult {
  imageUrl: string;
  artist: string;
  title: string;
  year?: string;
  similarity: number;
}

// Parse the caption string like "94.0 Diego Velazquez - A Young Man Self Portrait 1624"
function parseCaption(caption: string): { artist: string; title: string; year?: string; similarity: number } {
  const match = caption.match(/^([\d.]+)\s+(.+?)\s+-\s+(.+?)(?:\s+(\d{4}))?$/);
  if (match) {
    return {
      similarity: parseFloat(match[1]),
      artist: match[2].trim(),
      title: match[3].trim(),
      year: match[4],
    };
  }
  // Fallback parsing
  const parts = caption.split(' ');
  const similarity = parseFloat(parts[0]) || 0;
  const rest = parts.slice(1).join(' ');
  const dashIndex = rest.indexOf(' - ');
  if (dashIndex > 0) {
    return {
      similarity,
      artist: rest.substring(0, dashIndex).trim(),
      title: rest.substring(dashIndex + 3).trim(),
    };
  }
  return { similarity, artist: 'Unknown Artist', title: rest || 'Untitled' };
}

export default function SearchPage() {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<ArtworkResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Call Gradio API with uploaded image
  const searchSimilarArt = useCallback(async (file: File) => {
    setIsLoading(true);
    setError(null);
    setResults([]);

    try {
      // Step 1: Upload the file to Gradio
      const formData = new FormData();
      formData.append('files', file);

      const uploadResponse = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error('Failed to upload image');
      }

      const uploadedFiles = await uploadResponse.json();
      if (!uploadedFiles || uploadedFiles.length === 0) {
        throw new Error('No file path returned');
      }

      const uploadedPath = uploadedFiles[0];

      // Step 2: Submit prediction job
      const callResponse = await fetch(`${API_BASE_URL}/call/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: [{
            path: uploadedPath,
            orig_name: file.name,
            meta: { _type: 'gradio.FileData' }
          }]
        }),
      });

      if (!callResponse.ok) {
        throw new Error('Failed to submit prediction');
      }

      const callResult = await callResponse.json();
      const eventId = callResult.event_id;

      if (!eventId) {
        throw new Error('No event ID returned');
      }

      // Step 3: Get results via SSE
      const resultResponse = await fetch(`${API_BASE_URL}/call/predict/${eventId}`, {
        headers: { 'Accept': 'text/event-stream' },
      });

      if (!resultResponse.ok) {
        throw new Error('Failed to fetch results');
      }

      const text = await resultResponse.text();
      const lines = text.split('\n');

      let galleryData = null;
      for (const line of lines) {
        if (line.startsWith('data:')) {
          const dataStr = line.substring(5).trim();
          try {
            const parsed = JSON.parse(dataStr);
            if (Array.isArray(parsed)) {
              galleryData = parsed;
              break;
            }
          } catch {
            continue;
          }
        }
      }

      if (!galleryData || !galleryData[0]) {
        throw new Error('No results returned');
      }

      // Parse results (take top 4)
      const artworkResults: ArtworkResult[] = [];
      const items = galleryData[0].slice(0, 4);

      for (const item of items) {
        let imageUrl = '';
        let caption = '';

        if (typeof item === 'object' && item.image) {
          imageUrl = item.image.url || item.image.path || '';
          caption = item.caption || '';
        } else if (Array.isArray(item) && item.length >= 2) {
          imageUrl = item[0];
          caption = item[1];
        }

        if (!imageUrl) continue;

        // Fix URL format
        if (imageUrl.startsWith('/')) {
          imageUrl = `${API_BASE_URL}${imageUrl}`;
        }
        if (imageUrl.includes('/c/file=')) {
          imageUrl = imageUrl.replace('/c/file=', '/file=');
        }

        const parsed = parseCaption(caption);
        artworkResults.push({
          imageUrl,
          artist: parsed.artist,
          title: parsed.title,
          year: parsed.year,
          similarity: parsed.similarity,
        });
      }

      setResults(artworkResults);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (evt) => {
        setUploadedImage(evt.target?.result as string);
      };
      reader.readAsDataURL(file);
      searchSimilarArt(file);
    }
  }, [searchSimilarArt]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (evt) => {
        setUploadedImage(evt.target?.result as string);
      };
      reader.readAsDataURL(file);
      searchSimilarArt(file);
    }
  }, [searchSimilarArt]);

  const clearImage = () => {
    setUploadedImage(null);
    setResults([]);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-stone-950">
      {/* Navbar */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-stone-900/80 backdrop-blur-md border-b border-stone-700/50">
        <div className="max-w-7xl mx-auto px-8 py-4">
          <div className="flex items-center justify-center">
            <Link href="/" className="font-serif text-2xl text-stone-100 italic tracking-wide">
              img2art
            </Link>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="pt-24 pb-12 px-6">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="font-serif text-4xl md:text-5xl text-stone-100 mb-4 italic">
              Find Your Art Match
            </h1>
            <p className="text-stone-400 text-lg">
              Upload an image and we&apos;ll find similar artworks from our collection
            </p>
          </div>

          {/* Upload Area */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`
              relative border transition-all duration-300
              ${isDragging
                ? 'border-stone-400 bg-stone-800/50'
                : 'border-stone-700 hover:border-stone-500 bg-stone-900/50'
              }
              ${uploadedImage ? 'p-4' : 'p-16'}
            `}
          >
            {uploadedImage ? (
              <div className="relative">
                <div className="relative aspect-video max-h-96 mx-auto overflow-hidden">
                  <Image
                    src={uploadedImage}
                    alt="Uploaded image"
                    fill
                    className="object-contain"
                  />
                </div>
                <button
                  onClick={clearImage}
                  className="absolute top-2 right-2 p-2 bg-stone-900/80 hover:bg-stone-800 transition-colors border border-stone-700"
                >
                  <svg className="w-5 h-5 text-stone-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ) : (
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-6 border border-stone-700 flex items-center justify-center">
                  <svg className="w-8 h-8 text-stone-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <p className="text-stone-200 text-lg mb-2">
                  Drag and drop an image here
                </p>
                <p className="text-stone-500 text-sm mb-6">
                  or
                </p>
                <label className="inline-flex items-center gap-2 px-6 py-3 border border-stone-600 cursor-pointer hover:bg-stone-800 transition-colors">
                  <svg className="w-5 h-5 text-stone-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <span className="text-stone-200 tracking-wide">Browse files</span>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </label>
                <p className="text-stone-600 text-xs mt-4 tracking-wide">
                  Supports: JPG, PNG, WebP, GIF
                </p>
              </div>
            )}
          </div>

          {/* Loading State */}
          {isLoading && (
            <div className="mt-12 text-center">
              <div className="inline-block w-8 h-8 border-2 border-stone-400 border-t-transparent rounded-full animate-spin mb-4" />
              <p className="text-stone-400 italic">Searching for similar artworks...</p>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="mt-12 text-center">
              <p className="text-red-400">{error}</p>
            </div>
          )}

          {/* Results Grid */}
          {results.length > 0 && (
            <div className="mt-12">
              <h2 className="font-serif text-3xl text-stone-100 italic mb-8 text-center">
                Similar Artworks
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {results.map((result, index) => (
                  <div key={index} className="border border-stone-700 overflow-hidden group">
                    <div className="relative aspect-[4/5]">
                      <Image
                        src={result.imageUrl}
                        alt={result.title}
                        fill
                        className="object-cover"
                        unoptimized
                      />
                    </div>
                    <div className="p-4 bg-stone-900/80">
                      <h3 className="font-serif text-xl text-stone-100 italic mb-1">
                        {result.title}
                      </h3>
                      <p className="text-stone-400 text-sm">
                        {result.artist}
                        {result.year && <span className="text-stone-500"> • {result.year}</span>}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Info Section */}
          <div className="mt-16 grid md:grid-cols-3 gap-6">
            <div className="border border-stone-700 p-6">
              <div className="w-12 h-12 border border-stone-600 flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-stone-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="font-serif text-stone-100 text-xl mb-2 italic">Contextual Search</h3>
              <p className="text-stone-500 text-sm">Find art with similar themes, moods, and compositions</p>
            </div>
            <div className="border border-stone-700 p-6">
              <div className="w-12 h-12 border border-stone-600 flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-stone-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="font-serif text-stone-100 text-xl mb-2 italic">Expression Match</h3>
              <p className="text-stone-500 text-sm">Match portraits based on facial expressions and emotions</p>
            </div>
            <div className="border border-stone-700 p-6">
              <div className="w-12 h-12 border border-stone-600 flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-stone-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
                </svg>
              </div>
              <h3 className="font-serif text-stone-100 text-xl mb-2 italic">Shape & Form</h3>
              <p className="text-stone-500 text-sm">Discover art with similar shapes and visual structures</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
