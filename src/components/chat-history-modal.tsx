
'use client';

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { MessageSquare } from 'lucide-react';
import type { Conversation } from '@/app/page';

type ChatHistoryModalProps = {
  conversations: Conversation[];
  onSelectChat: (id: string) => void;
  isOpen: boolean;
  onOpenChange: (isOpen: boolean) => void;
  isLoading: boolean;
};

export function ChatHistoryModal({
  conversations,
  onSelectChat,
  isOpen,
  onOpenChange,
  isLoading,
}: ChatHistoryModalProps) {
  const handleSelect = (id: string) => {
    onSelectChat(id);
    onOpenChange(false); // Close modal on selection
  };

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="bg-background/80 backdrop-blur-sm sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Your Conversations</DialogTitle>
        </DialogHeader>
        <ScrollArea className="max-h-[60vh]">
          <div className="flex flex-col gap-2 p-4 pr-6">
            {conversations.length > 0 ? (
              conversations.map((convo) => (
                <Button
                  key={convo.id}
                  variant="ghost"
                  className="w-full justify-start gap-2"
                  onClick={() => handleSelect(convo.id)}
                  disabled={isLoading}
                >
                  <MessageSquare className="h-4 w-4" />
                  <span className="truncate">{convo.name || 'New Chat'}</span>
                </Button>
              ))
            ) : (
              <p className="text-center text-sm text-muted-foreground">
                No conversations yet.
              </p>
            )}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
