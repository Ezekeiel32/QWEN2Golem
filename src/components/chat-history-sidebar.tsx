
'use client';

import {
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarTrigger,
  useSidebar,
} from '@/components/ui/sidebar';
import { MessageSquare, MessagesSquare, Plus } from 'lucide-react';
import type { Conversation } from '@/app/page';
import { useIsMobile } from '@/hooks/use-mobile';
import { ThemeToggle } from './theme-toggle';
import { ChatHistoryModal } from './chat-history-modal';
import { useState } from 'react';

interface ChatHistorySidebarProps {
  conversations: Conversation[];
  activeChatId: string | null;
  onNewChat: () => void;
  onSelectChat: (id: string) => void;
  isLoading: boolean;
}

export function ChatHistorySidebar({
  conversations,
  activeChatId,
  onNewChat,
  onSelectChat,
  isLoading,
}: ChatHistorySidebarProps) {
  const isMobile = useIsMobile();
  const { state } = useSidebar();
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between p-4">
        <h4 className="text-sm font-medium">Aether AI™ (by ZPEDeepNet)</h4>
        <Button
          variant="ghost"
          size="icon"
          // The original code had SidebarTrigger here, but SidebarTrigger is not imported.
          // Assuming it was intended to be removed or replaced with a placeholder.
          // For now, I'm removing it as it's not available.
          // <SidebarTrigger />
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarMenu>
          <SidebarMenuItem className="p-2">
            <SidebarMenuButton
              onClick={onNewChat}
              disabled={isLoading}
              variant="default"
              className="w-full"
              tooltip={{
                children: 'New Chat',
                side: 'right',
                align: 'center',
              }}
            >
              <Plus />
              <span>New Chat</span>
            </SidebarMenuButton>
          </SidebarMenuItem>

          {/* New Button for collapsed view */}
          {state === 'collapsed' && !isMobile && (
             <SidebarMenuItem className="p-2">
              <SidebarMenuButton
                onClick={() => setIsModalOpen(true)}
                disabled={isLoading}
                variant="outline"
                className="w-full"
                tooltip={{
                  children: 'View Chats',
                  side: 'right',
                  align: 'center',
                }}
              >
                <MessagesSquare />
                <span className="sr-only">View Chats</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          )}

          {/* Existing chat list for expanded view */}
          {state === 'expanded' && (
            <div className="flex-1 space-y-1 p-2">
              {conversations.map((convo) => (
                <SidebarMenuItem key={convo.id}>
                  <SidebarMenuButton
                    isActive={activeChatId === convo.id}
                    onClick={() => onSelectChat(convo.id)}
                    disabled={isLoading}
                    tooltip={{
                      children: convo.name || 'New Chat',
                      side: 'right',
                      align: 'center',
                    }}
                  >
                    <MessageSquare />
                    <span className="truncate">{convo.name || 'New Chat'}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </div>
          )}
        </SidebarMenu>
      </SidebarContent>
      <SidebarFooter>
        <div className="flex w-full items-center justify-end p-2 group-data-[collapsible=icon]:justify-center">
            <ThemeToggle />
        </div>
      </SidebarFooter>

      {/* Render the modal */}
      <ChatHistoryModal
        conversations={conversations}
        onSelectChat={onSelectChat}
        isOpen={isModalOpen}
        onOpenChange={setIsModalOpen}
        isLoading={isLoading}
      />
    </div>
  );
}
