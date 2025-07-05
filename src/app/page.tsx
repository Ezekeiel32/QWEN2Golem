import { ChatPanel } from '@/components/chat-panel';

export default function Home() {
  return (
    <div className="flex h-svh w-full items-center justify-center bg-background p-4 sm:p-8">
      <ChatPanel />
    </div>
  );
}
