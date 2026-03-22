#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡易 Word 生成器 - 無需額外套件
直接讀取 Markdown 文件，轉換為 Word 格式
"""

def generate_word_using_markdown():
    """
    使用系統內建工具將 Markdown 轉換為 Word
    適用於 Windows 系統
    """
    import subprocess
    import os
    
    # 合併所有章節
    chapters = [
        '量子廣義相對論_修改版_第1-2章.md',
        '量子廣義相對論_修改版_第3-5章.md',
        '量子廣義相對論_修改版_第6-9章.md'
    ]
    
    # 合併內容
    full_content = []
    full_content.append('# 量子廣義相對論的符號模擬研究：基於計算輔助的理論探索\n')
    full_content.append('**作者：劉哲廷**\n')
    full_content.append('---\n\n')
    
    for chapter_file in chapters:
        if os.path.exists(chapter_file):
            with open(chapter_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 移除重複的標題
                if chapter_file != chapters[0]:
                    content = content.split('\n', 2)[-1]
                full_content.append(content)
                full_content.append('\n\n')
        else:
            print(f'⚠️  找不到文件：{chapter_file}')
    
    # 寫入合併文件
    merged_file = '量子廣義相對論_完整報告.md'
    with open(merged_file, 'w', encoding='utf-8') as f:
        f.writelines(full_content)
    
    print(f'✅ 已合併 Markdown 文件：{merged_file}')
    print()
    print('📋 轉換為 Word 的方法：')
    print()
    print('方法 1（推薦）：使用 Pandoc')
    print('  1. 下載 Pandoc：https://pandoc.org/installing.html')
    print(f'  2. 執行命令：pandoc {merged_file} -o 量子廣義相對論_劉哲廷.docx')
    print()
    print('方法 2：使用 Word 直接開啟')
    print(f'  1. 用 Microsoft Word 開啟 {merged_file}')
    print('  2. 另存為 .docx 格式')
    print()
    print('方法 3：使用線上轉換工具')
    print('  1. 訪問 https://www.markdowntoword.com/')
    print(f'  2. 上傳 {merged_file}')
    print('  3. 下載轉換後的 Word 文件')
    
    return merged_file


def create_rtf_document():
    """
    創建 RTF 格式文檔（可被 Word 開啟）
    不需要額外套件
    """
    
    chapters = [
        '量子廣義相對論_修改版_第1-2章.md',
        '量子廣義相對論_修改版_第3-5章.md',
        '量子廣義相對論_修改版_第6-9章.md'
    ]
    
    rtf_content = []
    rtf_content.append(r'{\rtf1\ansi\deff0')
    rtf_content.append(r'{\fonttbl{\f0\fnil\fcharset136 新細明體;}}')
    rtf_content.append(r'{\colortbl;\red0\green0\blue0;\red255\green0\blue0;}')
    rtf_content.append('\n')
    
    # 標題
    rtf_content.append(r'\qc\f0\fs32\b 量子廣義相對論的符號模擬研究：\line 基於計算輔助的理論探索\b0\par')
    rtf_content.append(r'\qc\fs28\b 作者：劉哲廷\b0\par')
    rtf_content.append(r'\par\par')
    
    # 讀取並轉換內容
    for chapter_file in chapters:
        try:
            with open(chapter_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                
                if not line:
                    rtf_content.append(r'\par')
                    continue
                
                # 處理標題
                if line.startswith('## 第'):
                    clean_line = line.replace('##', '').strip()
                    rtf_content.append(f'\\fs28\\b {clean_line}\\b0\\par')
                elif line.startswith('### '):
                    clean_line = line.replace('###', '').strip()
                    rtf_content.append(f'\\fs24\\b {clean_line}\\b0\\par')
                elif line.startswith('**') and line.endswith('**'):
                    clean_line = line.replace('**', '')
                    rtf_content.append(f'\\b {clean_line}\\b0\\par')
                elif not line.startswith('#') and not line.startswith('---'):
                    # 普通段落
                    # 處理粗體
                    clean_line = line.replace('**', '\\b ')
                    clean_line = clean_line.replace('**', '\\b0 ')
                    rtf_content.append(f'\\fs24 {clean_line}\\par')
                    
        except FileNotFoundError:
            print(f'⚠️  找不到文件：{chapter_file}')
            continue
    
    rtf_content.append(r'\par\par')
    rtf_content.append(r'\qc\cf2\i （本報告為個人理論探索，結果尚屬初步推測，需進一步驗證與討論）\i0\cf1\par')
    rtf_content.append('}')
    
    # 寫入 RTF 文件
    output_file = '量子廣義相對論_劉哲廷.rtf'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rtf_content))
    
    print(f'✅ RTF 文檔已生成：{output_file}')
    print('📝 可以用 Microsoft Word 直接開啟此文件')
    return output_file


if __name__ == '__main__':
    print('=' * 60)
    print('量子廣義相對論報告 - Word 文檔生成工具')
    print('=' * 60)
    print()
    
    print('選擇生成方式：')
    print()
    print('1. 生成合併的 Markdown 文件（需要手動轉換為 Word）')
    print('2. 生成 RTF 文件（可直接用 Word 開啟）')
    print()
    
    choice = input('請選擇 (1/2): ').strip()
    
    if choice == '1':
        generate_word_using_markdown()
    elif choice == '2':
        create_rtf_document()
    else:
        print('生成兩種格式...')
        print()
        print('[方法 1] 合併 Markdown:')
        generate_word_using_markdown()
        print()
        print('[方法 2] RTF 文檔:')
        create_rtf_document()
    
    print()
    print('✅ 完成！')

