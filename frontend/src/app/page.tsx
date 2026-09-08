"use client";
import { useEffect, useState } from 'react';
import Sidebar from '../components/Sidebar';
import Dashboard from '../components/Dashboard';
import AgentChat from '../components/AgentChat';
import type { Entity } from '../lib/types';

type Theme = 'dark' | 'light';

function initialTheme(): Theme {
  if (typeof window === 'undefined') return 'dark';
  const savedTheme = window.localStorage.getItem('theme');
  return savedTheme === 'light' || savedTheme === 'dark' ? savedTheme : 'dark';
}

export default function Home() {
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isChatOpen, setIsChatOpen] = useState(true);
  const [isChatExpanded, setIsChatExpanded] = useState(false);
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    const root = document.documentElement;
    const body = document.body;
    root.classList.remove('theme-dark', 'theme-light');
    body.classList.remove('theme-dark', 'theme-light');
    root.classList.add(theme === 'light' ? 'theme-light' : 'theme-dark');
    body.classList.add(theme === 'light' ? 'theme-light' : 'theme-dark');
    root.setAttribute('data-theme', theme);
    body.setAttribute('data-theme', theme);
    window.localStorage.setItem('theme', theme);
  }, [theme]);

  return (
    <main className="flex h-screen w-screen overflow-hidden bg-neutral-950 font-sans antialiased text-neutral-100 selection:bg-purple-900 selection:text-white">
      {/* Left Column: Sidebar */}
      <div className={`transition-all duration-300 ease-in-out flex flex-col border-r border-neutral-800 shrink-0 shadow-2xl z-20 ${isSidebarOpen ? 'w-[20%] min-w-[320px] max-w-[400px]' : 'w-0 overflow-hidden border-none opacity-0'}`}>
        <div className="flex justify-end p-2 border-b border-neutral-800/50 bg-neutral-950">
          <button onClick={() => setIsSidebarOpen(false)} className="text-neutral-500 hover:text-purple-400 p-1" title="Close Sidebar">❮</button>
        </div>
        <div className="flex-1 overflow-hidden">
          <Sidebar
            onSelectEntity={setSelectedEntity}
            onGlobalSearch={() => setSelectedEntity(null)}
          />
        </div>
      </div>

      {/* Center Column: Dashboard */}
      <div className="flex-1 overflow-hidden flex flex-col relative bg-neutral-950">
        <button
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          className="fixed bottom-4 left-4 z-40 bg-neutral-900 border border-neutral-800 text-purple-400 px-3 h-10 rounded-lg hover:bg-neutral-800 shadow-lg flex items-center justify-center gap-2 font-semibold text-sm"
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        >
          <span>{theme === 'dark' ? '☀' : '☾'}</span>
          <span>{theme === 'dark' ? 'Light' : 'Dark'}</span>
        </button>
        {!isSidebarOpen && (
          <button onClick={() => setIsSidebarOpen(true)} className="absolute top-4 left-4 z-30 bg-neutral-900 border border-neutral-800 text-purple-400 w-10 h-10 rounded-lg hover:bg-neutral-800 shadow-lg flex items-center justify-center font-bold" title="Open Sidebar">
            ☰
          </button>
        )}
        {!isChatOpen && (
          <button onClick={() => setIsChatOpen(true)} className="absolute top-4 right-4 z-30 bg-neutral-900 border border-neutral-800 text-purple-400 w-10 h-10 rounded-lg hover:bg-neutral-800 shadow-lg flex items-center justify-center font-bold" title="Open Chat">
            💬
          </button>
        )}
        <Dashboard entity={selectedEntity} setSelectedEntity={setSelectedEntity} />
      </div>

      {/* Right Column: Chat */}
      <div className={`transition-all duration-300 ease-in-out flex flex-col border-l border-neutral-800 shrink-0 shadow-2xl z-20 relative bg-neutral-950 ${isChatOpen ? (isChatExpanded ? 'w-[50%] min-w-[600px]' : 'w-[30%] min-w-[400px]') : 'w-0 overflow-hidden border-none opacity-0'}`}>
        <AgentChat
          entity={selectedEntity}
          isExpanded={isChatExpanded}
          onToggleExpand={() => setIsChatExpanded(!isChatExpanded)}
          onClose={() => setIsChatOpen(false)}
        />
      </div>
    </main>
  );
}
