
"use client";

import { ChatPanel } from '@/components/chat-panel';
import { ChatHistorySidebar } from '@/components/chat-history-sidebar';
import { useState, useEffect } from 'react';
import { golemChat } from '@/ai/flows/golem-chat';
import { useToast } from '@/hooks/use-toast';
import { v4 as uuidv4 } from 'uuid';
import {
  SidebarProvider,
  Sidebar,
  SidebarInset,
  SidebarHeader,
  SidebarContent,
  SidebarFooter,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarTrigger,
} from '@/components/ui/sidebar';
import { MessageSquare, MessagesSquare, Plus } from 'lucide-react';
import { ThemeToggle } from '@/components/theme-toggle';
import { Button } from '@/components/ui/button';
import { useSidebar } from '@/hooks/use-sidebar';
import { Separator } from '@/components/ui/separator';

// Force redeploy on Vercel
export type Message = {
  role: 'user' | 'assistant';
  content: string; // For user messages, this is the prompt. For assistant, the direct response.
  aetherAnalysis?: string | null;
  recommendation?: string | null;
  file?: {
    name: string;
  };
  golemStats?: any;
};

export type Conversation = {
  id: string; // This will now serve as the sessionId
  name: string;
  messages: Message[];
};

const SEFIROT_NAMES = [
  'Keter', 'Chokhmah', 'Binah', 'Chesed', 'Gevurah', 
  'Tiferet', 'Netzach', 'Hod', 'Yesod', 'Malkuth'
];

const SACRED_PHRASES = ["אמת", "חיים", "אור", "חכמה", "בינה", "דעת"];

// Create a separate component that uses the sidebar hook
function ChatApp() {
  const { toast } = useToast();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const { isMobile, state } = useSidebar();
  const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);

  // Golem State
  const [golemActivated, setGolemActivated] = useState(false);
  const [phraseClicks, setPhraseClicks] = useState<Record<string, number>>({});
  const [sefirotSettings, setSefirotSettings] = useState(() =>
    SEFIROT_NAMES.reduce((acc, name) => ({ ...acc, [name]: 0.5 }), {} as Record<string, number>)
  );

  useEffect(() => {
    try {
      const savedConversations = localStorage.getItem('aetherai-conversations');
      const savedActiveChatId = localStorage.getItem('aetherai-activeChatId');
      
      const loadedConversations = savedConversations ? JSON.parse(savedConversations) : [];
      setConversations(loadedConversations);

      if (savedActiveChatId && loadedConversations.some((c: Conversation) => c.id === savedActiveChatId)) {
        setActiveChatId(savedActiveChatId);
      } else if (loadedConversations.length > 0) {
        setActiveChatId(loadedConversations[0].id);
      } else {
        // Don't create a new chat on load, let the user do it.
      }
    } catch (error) {
      console.error("Failed to load state from localStorage", error);
      setConversations([]);
      setActiveChatId(null);
    } finally {
        setIsLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (isLoaded) {
      try {
        localStorage.setItem('aetherai-conversations', JSON.stringify(conversations));
        if (activeChatId) {
          localStorage.setItem('aetherai-activeChatId', activeChatId);
        } else {
          localStorage.removeItem('aetherai-activeChatId');
        }
      } catch (error) {
        console.error("Failed to save state to localStorage", error);
      }
    }
  }, [conversations, activeChatId, isLoaded]);

  const handleNewChat = () => {
    const newId = uuidv4();
    const newConversation: Conversation = {
      id: newId,
      name: 'New Chat',
      messages: [],
    };
    setConversations(prev => [newConversation, ...prev]);
    setActiveChatId(newId);
    return newId;
  };

  const handleSelectChat = (id: string) => {
    setActiveChatId(id);
  };

  const handleSendMessage = async (input: string, temperature: number, file: File | null) => {
    let currentChatId = activeChatId;

    if (!currentChatId) {
      currentChatId = handleNewChat();
    }
    
    if (!currentChatId) {
       console.error("No active chat ID even after trying to create a new one.");
       toast({
         variant: 'destructive',
         title: 'Error',
         description: 'Could not establish a chat session. Please refresh.',
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

    const originalConversations = JSON.parse(JSON.stringify(conversations));

    setConversations(prev =>
      prev.map(convo => {
        if (convo.id === currentChatId) {
          const newName = convo.messages.length === 0 && input.trim() ? input.substring(0, 40) + '...' : convo.name;
          return { ...convo, name: newName, messages: [...convo.messages, userMessage] };
        }
        return convo;
      })
    );
    setIsLoading(true);

    try {
      const activationPhrases = Object.entries(phraseClicks)
        .flatMap(([phrase, count]) => Array(count).fill(phrase));

      const result = await golemChat({
        prompt: input,
        sessionId: currentChatId,
        temperature,
        fileContent,
        golemActivated,
        activationPhrases,
        sefirotSettings,
      });

      if (result.directResponse) {
        const assistantMessage: Message = {
          role: 'assistant',
          content: result.directResponse,
          aetherAnalysis: result.aetherAnalysis,
          recommendation: result.recommendation,
          golemStats: result, // Pass the whole result object as golemStats
        };
        setConversations(prev =>
          prev.map(convo =>
            convo.id === currentChatId
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
      console.error('Error calling golemChat:', error);
       setConversations(originalConversations);
       const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred.';
       
       toast({
         variant: 'destructive',
         title: 'Error',
         description: `Failed to get response from Golem server: ${errorMessage}`,
      });
      
    } finally {
      setIsLoading(false);
    }
  };

  const activeChat = conversations.find(c => c.id === activeChatId);

  return (
    <>
      <Sidebar variant="inset" collapsible="icon">
        <SidebarHeader>
          <div className="flex w-full items-center justify-between p-4">
            <h4 className="text-sm font-medium">Aether AI™ (by ZPEDeepNet)</h4>
            <SidebarTrigger />
          </div>
        </SidebarHeader>
        <SidebarContent>
          <SidebarMenu>
            <SidebarMenuItem className="p-2">
              <SidebarMenuButton
                onClick={handleNewChat}
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
            <ChatHistorySidebar
              conversations={conversations}
              onSelectChat={handleSelectChat}
              isLoading={isLoading}
              isOpen={isHistoryModalOpen}
              onOpenChange={setIsHistoryModalOpen}
            />
          </SidebarMenu>
        </SidebarContent>
        <SidebarFooter>
          <div className="flex items-center justify-between p-4">
            <ThemeToggle />
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsHistoryModalOpen(true)}
              className="flex items-center gap-2"
            >
              <MessagesSquare className="h-4 w-4" />
              <span className="text-xs">History</span>
            </Button>
          </div>
        </SidebarFooter>
      </Sidebar>
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-[[data-collapsible=icon]]/sidebar-wrapper:h-12">
          <div className="flex items-center gap-2 px-4">
            <SidebarTrigger className="-ml-1" />
            <Separator orientation="vertical" className="mr-2 h-4" />
            <h1 className="text-lg font-semibold">Aether AI™</h1>
          </div>
        </header>
        <div className="flex flex-1 flex-col gap-4 p-4 pt-0">
          <div className="grid auto-rows-min gap-4 md:grid-cols-3">
            <div className="aspect-video rounded-xl bg-muted/50" />
            <div className="aspect-video rounded-xl bg-muted/50" />
            <div className="aspect-video rounded-xl bg-muted/50" />
          </div>
          <div className="min-h-[100vh] flex-1 rounded-xl bg-muted/50 md:min-h-min">
            <ChatPanel
              messages={activeChat?.messages || []}
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
              isChatSelected={!!activeChat}
              onNewChat={handleNewChat}
              golemActivated={golemActivated}
              setGolemActivated={setGolemActivated}
              phraseClicks={phraseClicks}
              setPhraseClicks={setPhraseClicks}
              sefirotSettings={sefirotSettings}
              setSefirotSettings={setSefirotSettings}
              sefirotNames={SEFIROT_NAMES}
            />
          </div>
        </div>
      </SidebarInset>
    </>
  );
}

export default function Home() {
  return (
    <SidebarProvider>
      <ChatApp />
    </SidebarProvider>
  );
}
