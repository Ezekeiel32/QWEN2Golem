"use client";

import { ChatPanel } from '@/components/chat-panel';
import { ChatHistorySidebar } from '@/components/chat-history-sidebar';
import { useState, useEffect } from 'react';
import type { Message } from '@/components/chat-message';
import { ollamaChat, type OllamaHistoryItem } from '@/ai/flows/ollama-chat';
import { useToast } from '@/hooks/use-toast';
import { v4 as uuidv4 } from 'uuid';
import { SidebarProvider, Sidebar, SidebarInset } from '@/components/ui/sidebar';

export type Conversation = {
  id: string;
  name: string;
  messages: Message[];
};

export default function Home() {
  const { toast } = useToast();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Select the first chat on initial load if it exists
    if (conversations.length > 0 && !activeChatId) {
      setActiveChatId(conversations[0].id);
    }
  }, [conversations, activeChatId]);

  const handleNewChat = () => {
    const newId = uuidv4();
    const newConversation: Conversation = {
      id: newId,
      name: 'New Chat',
      messages: [],
    };
    setConversations(prev => [newConversation, ...prev]);
    setActiveChatId(newId);
  };

  const handleSelectChat = (id: string) => {
    setActiveChatId(id);
  };

  const handleSendMessage = async (input: string, temperature: number, file: File | null) => {
    if (!activeChatId) {
      toast({
        variant: 'destructive',
        title: 'No active chat',
        description: 'Please select a chat or create a new one.',
      });
      return;
    }

    let fileContent: string | undefined = undefined;
    if (file) {
      try {
        fileContent = await file.text();
      } catch (error) {
        console.error("Error reading file:", error);
        toast({
            variant: 'destructive',
            title: 'File Read Error',
            description: 'Could not read the selected file.',
        });
        return;
      }
    }

    const userMessage: Message = {
      role: 'user',
      content: input,
      ...(file && { file: { name: file.name } })
    };

    setConversations(prev =>
      prev.map(convo => {
        if (convo.id === activeChatId) {
          const newName = convo.messages.length === 0 && input.trim() ? input.substring(0, 40) + '...' : convo.name;
          return { ...convo, name: newName, messages: [...convo.messages, userMessage] };
        }
        return convo;
      })
    );
    setIsLoading(true);

    try {
      const activeConversation = conversations.find(c => c.id === activeChatId);
      const history: OllamaHistoryItem[] = activeConversation
        ? activeConversation.messages.map(({ role, content }) => ({ role, content }))
        : [];
      
      const result = await ollamaChat({
        prompt: input,
        history,
        temperature,
        fileContent,
      });

      if (result.response) {
        const assistantMessage: Message = {
          role: 'assistant',
          content: result.response,
        };
        setConversations(prev =>
          prev.map(convo =>
            convo.id === activeChatId
              ? { ...convo, messages: [...convo.messages, assistantMessage] }
              : convo
          )
        );
      } else {
         toast({
          variant: 'default',
          title: 'Empty Response',
          description: 'The AI returned an empty response. Please try a different question.',
        });
      }
    } catch (error) {
      console.error('Error calling ollamaChat:', error);
      toast({
        variant: 'destructive',
        title: 'Error',
        description: 'Failed to get a response from the AI. Please try again.',
      });
      setConversations(prev =>
        prev.map(convo =>
          convo.id === activeChatId
            ? { ...convo, messages: convo.messages.slice(0, -1) }
            : convo
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const activeChat = conversations.find(c => c.id === activeChatId);

  return (
    <SidebarProvider>
      <Sidebar variant="inset" collapsible="icon">
        <ChatHistorySidebar
          conversations={conversations}
          activeChatId={activeChatId}
          onNewChat={handleNewChat}
          onSelectChat={handleSelectChat}
          isLoading={isLoading}
        />
      </Sidebar>
      <SidebarInset>
        <ChatPanel
          messages={activeChat?.messages ?? []}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          isChatSelected={!!activeChat}
          onNewChat={handleNewChat}
        />
      </SidebarInset>
    </SidebarProvider>
  );
}
