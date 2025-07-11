
'use client';

import {
  // Removed SidebarMenu, SidebarMenuItem, SidebarMenuButton as they are handled in page.tsx
  // useSidebar is no longer needed here
} from '@/components/ui/sidebar';
// All lucide-react imports are no longer needed here
// import { MessageSquare, MessagesSquare, Plus } from 'lucide-react';
import type { Conversation } from '@/app/page';
// useIsMobile is no longer needed here
// import { useIsMobile } from '@/hooks/use-mobile';
// ThemeToggle is no longer needed here
// import { ThemeToggle } from './theme-toggle';
import { ChatHistoryModal } from './chat-history-modal';
import { useState } from 'react';
// Button is no longer needed here
// import { Button } from './ui/button';

interface ChatHistorySidebarProps {
  conversations: Conversation[];
  onSelectChat: (id: string) => void;
  isLoading: boolean;
  isOpen: boolean;
  onOpenChange: (isOpen: boolean) => void;
}

export function ChatHistorySidebar({
  conversations,
  onSelectChat,
  isLoading,
  isOpen,
  onOpenChange,
}: ChatHistorySidebarProps) {

  return (
    <ChatHistoryModal
      conversations={conversations}
      onSelectChat={onSelectChat}
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      isLoading={isLoading}
    />
  );
}
