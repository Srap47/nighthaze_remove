/**
 * App — Root application component with routing.
 *
 * Sets up react-router for client-side navigation. Wraps all pages with
 * the Layout component (which provides Header and Footer). Two routes:
 *   - / → HomePage (upload, demo, results)
 *   - /about → AboutPage (project info)
 */

import { BrowserRouter, Route, Routes } from 'react-router-dom';

import { Layout } from './components/layout/Layout';
import { AboutPage } from './pages/AboutPage';
import { HomePage } from './pages/HomePage';

function App() {
  return (
    <BrowserRouter>
      {/* Layout provides sticky header, main container, and footer structure */}
      <Layout>
        {/* Client-side routing with react-router */}
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App
