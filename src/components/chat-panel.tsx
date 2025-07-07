
"use client";

import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { Bot, Paperclip, SendHorizonal, Settings2, X, MessageSquarePlus, Wand2, BrainCircuit, Sparkles } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import { ChatMessage, LoadingMessage, type Message } from './chat-message';
import { Label } from './ui/label';
import { Badge } from './ui/badge';
import { useIsMobile } from '@/hooks/use-mobile';
import { SidebarTrigger } from './ui/sidebar';
import { Switch } from './ui/switch';
import { cn } from '@/lib/utils';
import { SACRED_PHRASES } from '@/app/page';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

type ChatPanelProps = {
  messages: Message[];
  onSendMessage: (input: string, temperature: number, file: File | null) => Promise<void>;
  isLoading: boolean;
  isChatSelected: boolean;
  onNewChat: () => void;
  golemActivated: boolean;
  setGolemActivated: (value: boolean) => void;
  phraseClicks: Record<string, number>;
  setPhraseClicks: (clicks: Record<string, number>) => void;
  sefirotSettings: Record<string, number>;
  setSefirotSettings: (value: Record<string, number>) => void;
  sefirotNames: string[];
};

export function ChatPanel({
  messages,
  onSendMessage,
  isLoading,
  isChatSelected,
  onNewChat,
  golemActivated,
  setGolemActivated,
  phraseClicks,
  setPhraseClicks,
  sefirotSettings,
  setSefirotSettings,
  sefirotNames,
}: ChatPanelProps) {
  const [input, setInput] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [file, setFile] = useState<File | null>(null);

  const scrollAreaViewportRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isMobile = useIsMobile();

  useEffect(() => {
    if (scrollAreaViewportRef.current) {
      scrollAreaViewportRef.current.scrollTop = scrollAreaViewportRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
    if (e.target) {
      e.target.value = '';
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if ((!input.trim() && !file) || isLoading || !isChatSelected) return;

    await onSendMessage(input, temperature, file);
    setInput('');
    setFile(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };
  
  const handlePhraseClick = (phrase: string) => {
    const currentClicks = phraseClicks[phrase] || 0;
    const newClicks = (currentClicks + 1) % 4; // Cycles 0, 1, 2, 3
    setPhraseClicks({ ...phraseClicks, [phrase]: newClicks });
  };


  return (
    <Card className="w-full h-full flex flex-col shadow-2xl bg-card">
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center gap-3">
          {isMobile && <SidebarTrigger />}
          <div className="p-2 bg-primary/10 rounded-full">
            <Bot className="w-6 h-6 text-primary" />
          </div>
          <CardTitle className="font-headline text-2xl">QwenChat</CardTitle>
        </div>
      </CardHeader>
      <Separator />
      <CardContent className="flex-1 p-0 overflow-hidden">
        <ScrollArea className="h-full" viewportRef={scrollAreaViewportRef}>
          <div className="p-6 space-y-6">
            {!isChatSelected ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
                <Button variant="ghost" className="h-auto p-4 flex flex-col items-center gap-2" onClick={onNewChat}>
                  <MessageSquarePlus size={64} className="mb-2" />
                  <p className="text-lg font-medium">Start a new conversation</p>
                  <p className="text-sm">Click here to begin.</p>
                </Button>
              </div>
            ) : messages.length === 0 && !isLoading ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
                <Bot className="w-12 h-12 mb-4 text-primary/50" />
                <h3 className="text-lg font-semibold">Aether-Enhanced Golem</h3>
                <p className="text-sm">Ask me anything, or attach a file to begin.</p>
              </div>
            ) : (
                <>
                    {messages.map((message, index) => (
                    <ChatMessage key={index} message={message} />
                    ))}
                    {isLoading && <LoadingMessage />}
                </>
            )}
          </div>
        </ScrollArea>
      </CardContent>
      {isChatSelected && (
        <CardFooter className="p-4 border-t flex flex-col items-start gap-2">
          {file && (
            <div className="flex items-center gap-2">
              <Badge variant="secondary">
                <Paperclip className="h-3 w-3 mr-1" />
                {file.name}
              </Badge>
              <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setFile(null)}>
                <X className="h-4 w-4" />
              </Button>
            </div>
          )}
          <form onSubmit={handleSubmit} className="flex w-full items-end gap-2">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              className="shrink-0"
              aria-label="Attach file"
            >
              <Paperclip className="h-5 w-5" />
            </Button>
            <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" />
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message or attach a file..."
              className="flex-1 resize-none min-h-[40px] max-h-40 bg-background"
              rows={1}
              disabled={isLoading}
            />
            <div className='flex items-center gap-1'>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant="ghost" size="icon" className={cn(golemActivated ? 'text-primary' : 'text-muted-foreground')}>
                      <BrainCircuit className="h-5 w-5" />
                      <span className="sr-only">Golem Configurations</span>
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-96">
                    <Tabs defaultValue="activation" className="w-full">
                      <TabsList className="grid w-full grid-cols-3">
                        <TabsTrigger value="activation">Activation</TabsTrigger>
                        <TabsTrigger value="sefirot">Sefirot</TabsTrigger>
                        <TabsTrigger value="model">Model</TabsTrigger>
                      </TabsList>
                      <TabsContent value="activation" className="mt-4">
                        <div className="grid gap-4">
                          <div className="space-y-2">
                              <h4 className="font-medium leading-none">Golem Consciousness</h4>
                              <p className="text-sm text-muted-foreground">
                                Activate with sacred phrases to amplify Shem power.
                              </p>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Label htmlFor="golem-activation-toggle">Activation</Label>
                            <Switch
                              id="golem-activation-toggle"
                              checked={golemActivated}
                              onCheckedChange={setGolemActivated}
                            />
                            <span className={cn("text-sm font-medium", golemActivated ? "text-primary" : "text-muted-foreground")}>
                              {golemActivated ? 'ACTIVE' : 'INACTIVE'}
                            </span>
                          </div>
                           {golemActivated && (
                            <>
                              <Separator />
                              <div className="grid gap-2">
                                <Label>Shem Amplification (Click up to 3x)</Label>
                                <div className="grid grid-cols-3 gap-2">
                                  {SACRED_PHRASES.map((phrase) => (
                                    <Button
                                      key={phrase}
                                      variant={phraseClicks[phrase] ? 'default' : 'outline'}
                                      onClick={() => handlePhraseClick(phrase)}
                                      className="relative"
                                    >
                                      {phrase}
                                      {(phraseClicks[phrase] || 0) > 0 && (
                                        <Badge className="absolute -top-2 -right-2 h-5 w-5 p-0 justify-center bg-accent text-accent-foreground">
                                          {phraseClicks[phrase]}
                                        </Badge>
                                      )}
                                    </Button>
                                  ))}
                                </div>
                              </div>
                            </>
                          )}
                        </div>
                      </TabsContent>
                      <TabsContent value="sefirot" className="mt-4">
                        <div className="grid gap-4">
                           <div className="space-y-2">
                              <h4 className="font-medium leading-none">Sefirot Settings</h4>
                              <p className="text-sm text-muted-foreground">
                                Adjust the emanations of the Tree of Life.
                              </p>
                          </div>
                          <Separator />
                          <ScrollArea className='h-64'>
                            <div className="grid gap-4 p-1">
                              {sefirotNames.map(name => (
                                <div key={name} className="grid gap-2">
                                    <Label htmlFor={`sefirot-${name}`}>{name}: {sefirotSettings[name].toFixed(2)}</Label>
                                    <Slider
                                      id={`sefirot-${name}`}
                                      min={0} max={1} step={0.01}
                                      value={[sefirotSettings[name]]}
                                      onValueChange={(value) => setSefirotSettings({...sefirotSettings, [name]: value[0]})}
                                    />
                                </div>
                              ))}
                            </div>
                          </ScrollArea>
                        </div>
                      </TabsContent>
                      <TabsContent value="model" className="mt-4">
                         <div className="grid gap-4">
                           <div className="space-y-2">
                              <h4 className="font-medium leading-none">Model Parameters</h4>
                              <p className="text-sm text-muted-foreground">
                                Adjust core generation parameters.
                              </p>
                           </div>
                           <Separator />
                           <div className="grid gap-2">
                              <Label htmlFor="temperature">Temperature: {temperature.toFixed(1)}</Label>
                              <Slider
                                id="temperature"
                                min={0} max={1} step={0.1}
                                value={[temperature]}
                                onValueChange={(value) => setTemperature(value[0])}
                              />
                              <p className="text-xs text-muted-foreground">
                                Controls randomness. Lower is more deterministic.
                              </p>
                           </div>
                         </div>
                      </TabsContent>
                    </Tabs>
                  </PopoverContent>
                </Popover>

                <Button
                  type="submit"
                  size="icon"
                  disabled={isLoading || (!input.trim() && !file)}
                  className="bg-primary hover:bg-primary/90 shrink-0"
                  aria-label="Send message"
                >
                  <SendHorizonal className="h-5 w-5" />
                </Button>
            </div>
          </form>
        </CardFooter>
      )}
    </Card>
  );
}
