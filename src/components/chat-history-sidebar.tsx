
'use client';

import { Button } from '@/components/ui/button';
import { MessageSquare, Plus } from 'lucide-react';
import type { Conversation } from '@/app/page';
import { ScrollArea } from './ui/scroll-area';

type ChatHistorySidebarProps = {
  conversations: Conversation[];
  activeChatId: string | null;
  onNewChat: () => void;
  onSelectChat: (id: string) => void;
  isLoading: boolean;
};

export function ChatHistorySidebar({
  conversations,
  activeChatId,
  onNewChat,
  onSelectChat,
  isLoading,
}: ChatHistorySidebarProps) {
  return (
    <aside className="flex h-full w-full max-w-[280px] flex-col border-r bg-card p-2">
      <div className="p-2">
        <Button onClick={onNewChat} className="w-full" disabled={isLoading}>
          <Plus className="mr-2 h-4 w-4" />
          New Chat
        </Button>
      </div>
      <ScrollArea className="flex-1">
        <div className="space-y-1 p-2">
            {conversations.map((convo) => (
            <Button
                key={convo.id}
                variant={activeChatId === convo.id ? 'secondary' : 'ghost'}
                className="w-full justify-start truncate"
                onClick={() => onSelectChat(convo.id)}
                disabled={isLoading}
            >
                <MessageSquare className="mr-2 h-4 w-4 flex-shrink-0" />
                <span className="truncate">{convo.name || 'New Chat'}</span>
            </Button>
            ))}
        </div>
      </ScrollArea>
    </aside>
  );
}
