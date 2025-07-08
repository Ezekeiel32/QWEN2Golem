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
import { MessageSquare, Plus } from 'lucide-react';
import type { Conversation } from '@/app/page';
import { useIsMobile } from '@/hooks/use-mobile';
import { ThemeToggle } from './theme-toggle';

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
  const isMobile = useIsMobile();
  const { state } = useSidebar();

  return (
    <>
      <SidebarHeader>
        <div className="flex w-full items-center justify-between">
          <h2 className="font-headline text-lg group-data-[collapsible=icon]:hidden">
            QwenChats
          </h2>
          {!isMobile && <SidebarTrigger />}
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
    </>
  );
}
